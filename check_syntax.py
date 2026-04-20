import ast, sys
files = ["src/flow_map.py", "src/detector.py", "src/config.py", "src/judge.py", "src/congestion_judge.py", "src/feature_extractor.py"]
for f in files:
    try:
        ast.parse(open(f, encoding="utf-8").read())
        print(f"{f} OK")
    except SyntaxError as e:
        print(f"{f} ERROR: {e}")
        sys.exit(1)
