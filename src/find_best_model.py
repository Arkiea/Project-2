import os.path
import glob
import json

import pandas as pd

def main():
    root_dir = os.path.join(os.path.dirname(__file__), '..')
    result_dir = os.path.join(root_dir, 'result')
    profitable = []
    for f in glob.glob(os.path.join(result_dir, '*.csv')):
        df = pd.read_csv(f)
        cum_profit = (1 + df['model returns']).cumprod().values[-1]
        if cum_profit > 1:
            profitable.append((f[14:], cum_profit))
    # with open('profitable.json', 'w') as f:
    #     f.write(json.dumps(profitable), indent=4)
    profitable_df = pd.DataFrame(profitable)
    profitable_df.columns = ['ta_indicators', 'cumulative_profit']
    profitable_df.sort_values(by=['cumulative_profit'], ascending=False, inplace=True)
    profitable_df.to_csv('profitable.csv', index=False)

if __name__ == '__main__':
    main()
