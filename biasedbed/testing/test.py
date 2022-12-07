import os
import pandas as pd
import wandb

from pingouin import friedman
import scikit_posthocs as sp

from omegaconf import OmegaConf


class HypothesesTesting:

    def __init__(self, wandbrun: str):
        self.run = wandbrun
        self.cfg = OmegaConf.load(os.path.join(os.path.dirname(__file__), 'config.yaml'))
        self.runs = self._download_runs()

    @staticmethod
    def _iterate_datafield(field, list_):
        # print(field, list_)
        if type(field) == dict:
            key = list(field.keys())[0]
            list_.append(key)
            new_list = HypothesesTesting._iterate_datafield(field[key], list_)
            return new_list
        elif type(field) == list:
            if len(field) == 1:
                new_list = HypothesesTesting._iterate_datafield(field[0], list_.copy())
                return new_list
            else:
                new_list = list()
                for f in field:
                    new_list.append(HypothesesTesting._iterate_datafield(f, list_.copy()))
                return new_list
        else:
            list_.append(field)
            return list_

    @staticmethod
    def _nested_get(dic, keys):
        try:
            for key in keys:
                dic = dic[key]
            return dic
        except KeyError:
            return 'None'

    def generate_individual_tables(self):
        # Grouping
        groups = list()
        for group in self.cfg.groups:
            group_fields = HypothesesTesting._iterate_datafield(OmegaConf.to_container(group), list())
            group_fields = [group_fields] if type(group_fields[0]) != list else group_fields
            for group_field in group_fields:
                groups.append('.'.join(group_field))
        print(groups)
        runs_grouped = self.runs.groupby(groups)
        for runs in runs_grouped:
            key = runs[0]
            results = runs[1]
            runs_latex = list()
            if not self.cfg.filters.last_only and not self.cfg.filters.best_only and not self.cfg.filters.best_validation:
                results_grouped_mean = results.groupby('run_id').mean() * 100
                results_grouped_mean = results_grouped_mean.drop('val/acc', axis=1)

                results_grouped_std = results.groupby('run_id').std() * 100
                results_grouped_std = results_grouped_std.drop('val/acc', axis=1)

                results_grouped_mean = results_grouped_mean.round(1)
                results_grouped_std = results_grouped_std.round(1)
                results = pd.concat([results_grouped_mean[col].astype(str) + " $\\pm$ " + results_grouped_std[col].astype(str) for col in results_grouped_mean], axis="columns")
                for i, run in enumerate(results.index):
                    vals = results.iloc[i].values.tolist()
                    runs_latex.append(f'\t {i+1} & {" & ".join(vals)} \\\\')
            else:
                results = results.drop(['run_id', 'algorithm.name', 'algorithm.optimizer', 'training.dataset.name'], axis=1)
                results = results.drop('val/acc', axis=1)
                for i, run in enumerate(results.index):
                    vals = results.iloc[i].values.tolist()
                    runs_latex.append(f'\t {i+1} & {" & ".join(str(round(val*100, 1)) for val in vals)} \\\\')
            datasets = results.keys().values
            latex = f"""
\\begin{{table}}[H]
 \\centering
 \\caption{{{key}}}
 \\begin{{tabular}}{{@{{}}l{''.join(['c'] * len(datasets))}@{{}}}}
   \\toprule
    Run & {' & '.join([f'{dataset.split("/")[1]}' for dataset in datasets])} \\\\
   \\midrule
   {chr(10).join(runs_latex)}
   \\bottomrule
  \\end{{tabular}}
\\end{{table}}
                            """
            print(latex)

    def generate_comparison(self) -> (pd.DataFrame, pd.DataFrame):
        # Grouping
        groups = list()
        for group in self.cfg.groups:
            group_fields = HypothesesTesting._iterate_datafield(OmegaConf.to_container(group), list())
            group_fields = [group_fields] if type(group_fields[0]) != list else group_fields
            for group_field in group_fields:
                groups.append('.'.join(group_field))
        runs_mean = self.runs.groupby(groups).mean() * 100
        runs_mean = runs_mean.drop('val/acc', axis=1)
        runs_std = self.runs.groupby(groups).std()*100
        runs_std = runs_std.drop('val/acc', axis=1)

        return runs_mean, runs_std

    def generate_latex_from_comparsion_table(self, title: str = 'Results choosing best performance over 100 epochs per run.'):
        table = self.compute_metrics()
        table.index = table.index.droplevel('algorithm.optimizer')
        table = table.set_index(table.index.to_flat_index())
        algorithms = table.index.values
        algorithms_latex = list()
        for i, algorithm in enumerate(algorithms):
            algorithms_latex.append(f'{algorithm[0]} ({algorithm[1]}) & {" & ".join(table.iloc[i].values.tolist())} \\\\')
        datasets = table.keys().values
        latex = f"""
\\begin{{table*}}
 \\centering
 \\caption{{{title}}}
 \\label{{tab:example}}
 \\begin{{tabular}}{{@{{}}l{''.join(['c']*len(datasets))}@{{}}}}
   \\toprule
    Algorithm & {' & '.join([f'{dataset.split("/")[1]}' for dataset in datasets])} \\\\
   \\midrule
   {chr(10).join(algorithms_latex)}
   \\bottomrule
  \\end{{tabular}}
\\end{{table*}}
                """
        return latex

    def compute_friedman(self):
        run_mean, _ = self.generate_comparison()
        table = run_mean #.drop(['test/ImageNet1k/acc'], axis=1)
        # print(table.keys())
        table.index = table.index.droplevel('algorithm.optimizer')
        table = table.set_index(table.index.to_flat_index()).transpose()
        table = table[[('ERM', 'ImageNet1k')] + [col for col in table.columns if col != ('ERM', 'ImageNet1k')]]
        stats = friedman(table, method='f')
        post_hoc_stats = None
        if stats['p-unc'].values[0] < 0.05:
            print('p < a --> applying post-hoc test.')
            post_hoc_stats = sp.posthoc_nemenyi_friedman(table)  # sp.posthoc_quade(table.transpose(), p_adjust='hommel')
        return stats, post_hoc_stats

    def generate_latex_from_friedman(self):
        stats, post_hoc_stats = self.compute_friedman()
        if post_hoc_stats is None:
            return stats
        # Move ERM to top


        algorithms = post_hoc_stats.index.values
        algorithms_latex = list()
        for i, algorithm in enumerate(algorithms):
            # print(algorithm)
            # print(post_hoc_stats.iloc[i].values.tolist())
            algorithms_latex.append(
                f'{algorithm[0]} ({algorithm[1]}) & {" & ".join([str(round(val, 3)) for val in post_hoc_stats.iloc[i].values.tolist()])} \\\\')
        latex = f"""
\\begin{{table*}}
 \\centering
 \\caption{{Post-hoc Nemenyi test.}}
 \\label{{tab:example}}
 \\begin{{tabular}}{{@{{}}l{''.join(['c'] * len(algorithms))}@{{}}}}
   \\toprule
    Algorithm & {' & '.join([f'{algorithm[0]} ({algorithm[1]})' for algorithm in algorithms])} \\\\
   \\midrule
   {chr(10).join(algorithms_latex)}
   \\bottomrule
  \\end{{tabular}}
\\end{{table*}}
                """
        return latex

    def compute_metrics(self):
        runs_mean, runs_std = self.generate_comparison()
        runs_mean = runs_mean.round(1)
        runs_std = runs_std.round(1)
        # runs = runs_mean.astype(str) + " $\\pm$  " + runs_std.astype(str)
        runs = pd.concat([runs_mean[col].astype(str) + " $\\pm$ " + runs_std[col].astype(str) for col in runs_mean], axis="columns")
        # runs = pd.merge(runs_mean, runs_std, on=groups, suffixes=('_mean', '_std'))
        # combinbe_values = lambda s1, s2: f"{s1} $\\pm$ {s2}"
        # runs = runs_mean.astype(str).combine(runs_std.astype(str), combinbe_values)
        return runs

    def _download_runs(self) -> pd.DataFrame:
        """
        :return:
        """
        api = wandb.Api()

        # Project is specified by <entity/project-name>
        runs = api.runs(self.run)

        dfs = list()
        for run in runs:
            df = run.history(pandas=True)
            df['run_id'] = run.id
            for datafield in self.cfg.datafields:
                field_list = HypothesesTesting._iterate_datafield(OmegaConf.to_container(datafield), list())
                field_list = [field_list] if type(field_list[0]) != list else field_list
                for field in field_list:
                    df['.'.join(field)] = HypothesesTesting._nested_get(run.config, field)
            dfs.append(df)

        runs = pd.concat(dfs, axis=0, ignore_index=True)

        # Filtering
        if self.cfg.filters.last_only:
            runs = runs.loc[runs.groupby('run_id')['_step'].idxmax()]
        elif self.cfg.filters.best_only:
            runs = runs.groupby('run_id').max()
        elif self.cfg.filters.best_validation:
            runs = runs.loc[runs.groupby('run_id')['val/acc'].idxmax()]
        elif self.cfg.filters.epochs_greater_equal:
            runs = runs[runs['_step'] >= self.cfg.filters.epochs_greater_equal]

        return runs.drop(['_step', '_runtime', 'val/loss', 'train/loss', 'train/acc', '_timestamp', 'train/loss_style', 'train/loss_adv'], axis=1)


if __name__ == "__main__":
    test = HypothesesTesting(wandbrun='biasedbed/main')
    # test.generate_comparison_table()
    test.generate_individual_tables()
    # stats, post_hoc_stats = test.compute_friedman()
    # print(test.generate_latex_from_friedman())