import json
from pathlib import Path

path = Path(
    'module04-ML-coding:Suprivised-learing/ensemble_model/bagging_mycode.ipynb')
nb = json.loads(path.read_text())

if len(nb['cells']) < 25:
    raise SystemExit('Unexpected notebook structure; cells count < 25')

nb['cells'][20]['source'] = [
    '# function for feature sampling\n',
    '\n',
    'def sample_features(df, percent):\n',
    '    cols = random.sample(df.columns.tolist()[:-1], int(percent * (df.shape[1] - 1)))\n',
    "    new_df = df.loc[:, cols + ['target']].copy()\n",
    '    return new_df\n',
]

nb['cells'][21]['source'] = [
    'sample_features(df, 0.2).head()\n',
]

nb['cells'][22]['source'] = [
    'def combined_samples(df, row_percent, col_percent):\n',
    '    sampled_rows = sample_rows(df, row_percent)\n',
    '    return sample_features(sampled_rows, col_percent)\n',
]

nb['cells'][23]['source'] = [
    '# Create a sampled dataset using row and column sampling\n',
    'df1 = combined_samples(df, 0.5, 0.5)\n',
    'df1.head()\n',
]

nb['cells'][24]['source'] = [
    'print(type(df1))\n',
    'print(df1.shape)\n',
    'df1.head()\n',
]

path.write_text(json.dumps(nb, indent=1))
print('patched')
