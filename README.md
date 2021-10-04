# Compiler for Nintendo EventFlow flowcharts

Tool to compile a readable, code-like format into eventflow flowcharts (bfevfl) for *Animal Crossing: New Horizons*.

This project is the counterpart of [acnh-eventflow-decompiler](https://github.com/asteriation/acnh-eventflow-decompiler).

This project compiles all 562/562 of the decompiled flows from ACNH v1.11.0, with 558/562 compiled flows
being smaller, and 3/562 being the same size when using `--optimize`.

## Usage

The decompiler may be run through `main.py` using Python 3.7+.

You will need to supply a `functions.csv` file containing typing information for EventFlow functions; this can be done for ACNH by downloading the appropriate 'functions.csv' sheet for your game version from [this spreadsheet](https://docs.google.com/spreadsheets/d/1AYM-UeRkbJuGy_nKv7AMngevwBtMdZPtfoHEQev8BhM/edit) as a CSV file.

You will also need to supply the evfl files to be compiled.

```bash
mkdir -p out/

python3 main.py --functions functions.csv \
                -d output_directory \
                file1.evfl file2.evfl ...
```

This outputs the compiled bfevfls into `output_directory`.

If compiling only a single file, `-o` can be used instead of `-d` to specify the output file name.

There are some flags starting with `--f` to optimize for output file size (at the expense of a
readable decompile), and `--optimize` enables them all.

## License

This software is licensed under the terms of the MIT License.
