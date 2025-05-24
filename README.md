git clone <repo URl>
pip install -r requirements.txt

capstone/
├── data/                    # OHLCV and trade CSV files
├── hmm.py      # Fit HMM and plot regimes
├── confusion_matrix.py # Compute confusion matrix
├── backtest_hmm.py # Sharpe, Sortino, Drawdown, Calmar
├── requirements.txt         #  dependencies
└── README.md                # Project documentation



 1) python script_scanner.py  #saves relevant file with required timeframe

 2) python ohlcv_converter.py data/"DIR_trades.csv" --interval 1s #converts it into ohlcv

 3) python hmm.py data/btc_usd_5m.csv --states 3 --iter 100 

 4) python confusion_matrix.py \  btc_usd_5m_hmm.csv \
  --pred-col sem \
  --true-col true_sem \
  --labels Bull Bear

5) python backtest_hmm.py \  btc_usd_5m_hmm.csv \                           
  --signal-col sem \
  --long Bull \
  --short Bear \
  --price-col close \
  --ppy 365 \
  --rf 0.0001