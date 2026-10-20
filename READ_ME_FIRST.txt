FILES
  access.tex                 manuscript (IEEE Access template). Compile twice: latexmk -g -pdf access.tex
  refs.bib                   references
  mehfo*.png                 your figures (regenerate without the embedded titles before submission)
  tables/                    REAL results generated from train.csv (1,458,644 rows) by run_all.py
  results_log_real_data.txt  console output of that run (evidence; includes the CHECK blocks)
  run_all.py, extended_experiments.py, run_in_colab.ipynb   code that produced tables/

TO REPRODUCE ON YOUR MACHINE (recommended once, so the hardware table and timings are yours):
  pip install pandas numpy scipy scikit-learn
  python run_all.py --csv train.csv --out tables
Everything except runtimes/hardware should come out identical (random seeds are fixed).
Do NOT use the --synthetic flag for anything you submit: it makes fake data for testing only.
