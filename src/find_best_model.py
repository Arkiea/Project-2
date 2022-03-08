import os.path
import json

import pandas as pd
import matplotlib.pyplot as plt

def main():
    root_dir = os.path.join(os.path.dirname(__file__), '..')
    result_dir = os.path.join(root_dir, 'result')
    profitable = []
    combinations = tuple(x.replace('.csv', '') for x in os.listdir(result_dir) if x.endswith('.csv'))
    for combination in combinations:
        df = pd.read_csv(os.path.join(result_dir, f"{combination}.csv"), index_col='date')
        name = '_'.join(sorted(df.columns[:3]))
        file_path_img = os.path.join(result_dir, f"{name}.png")
        title = ' | '.join(sorted(df.columns[:3]))
        cum_profit_btc = (1 + df['daily returns']).cumprod().values[-1]
        cum_profit = (1 + df['model returns']).cumprod().values[-1]
        # if the model outperforms hodling BTC
        if cum_profit > cum_profit_btc:
            profitable.append((combination, cum_profit))
            if not os.path.isfile(file_path_img):
                # plot the cumulative returns
                fig = (1 + df[['daily returns', 'model returns']]).cumprod().plot(title=title).get_figure()
                # save the plot as an image
                fig.savefig(file_path_img, bbox_inches='tight')
                # close all figures
                plt.close(fig)
    profitable_df = pd.DataFrame(profitable)
    profitable_df.columns = ['ta_indicators', 'cumulative_profit']
    profitable_df.set_index('ta_indicators', inplace=True)

    accuracy = []
    for combination in combinations:
        with open(os.path.join(result_dir, f"{combination}.json"), 'r') as json_file:
            result = json.load(json_file)
        if result['accuracy'] > 0.5:
            accuracy.append((combination, result['accuracy']))
    accuracy_df = pd.DataFrame(accuracy)
    accuracy_df.columns = ['ta_indicators', 'accuracy']
    accuracy_df.set_index('ta_indicators', inplace=True)
    result_df = pd.merge(profitable_df, accuracy_df, 'outer', left_index=True, right_index=True)
    result_df.sort_values(by=['cumulative_profit', 'accuracy'], ascending=False, inplace=True)
    result_df.to_csv('result.csv')


if __name__ == '__main__':
    main()
