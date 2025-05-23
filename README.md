1. Install Requirements(dependencies) using : pip install -r requirements.txt
2. Make sure to download English Web Treebank and keep it in data/raw_data/..(all three conllu files) here
3. Preprocessing : Execute: python preprocess.py
4. Making sure to validate data by ensuring all samples are having valid indices: Execute: python validate_data.py
5. Train the parser: Execute: python trainer.py --config config.json
