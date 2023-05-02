# tgca-ccaim
Environment constructed with Docker (see `env/Dockerfile`). Adjust `docker-compose.yml` as necessary.

Tasks are executed in `notebooks/Part 1.ipynb` and `notebooks/Part 2.ipynb`. Please note that due to file size, the raw data is not uploaded here. 
Please add a `data/` folder, with a `preprocessed/` and `raw` subfolder to the project. It is only needed to upload the raw `tcga.csv` file in `raw/` folder. `Part 1.ipynb` will take care of `tgca_preprocessed.csv`

The file tree structure should be as follows:
```
├── data
│   ├── preprocessed
│   │   └── tgca_preprocessed.csv
│   └── raw
│       └── tgca.csv
├── env
│   ...
├── notebooks
│   ...
└── src
    ...
```

