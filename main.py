import argparse
from pathlib import Path
import numpy as np
from src.data import load_data
from src.models import make_ridge
from src.features import get_features
from src.preprocessing import fit_transform_pair
from src.evaluation import metrics

ROOT=Path(__file__).resolve().parent

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest="command",required=True)
    sub.add_parser("validate-data",help="Validar estructura y objetivo del Parquet; no entrena")
    sub.add_parser("smoke",help="Ejecutar componentes con datos sintéticos")
    args=parser.parse_args()
    if args.command=="validate-data":
        frame=load_data(ROOT/"data/processed/df_long.parquet")
        print(f"Datos válidos: {len(frame)} carreras, {frame.id.nunique()} corredores; sin evaluación.")
    else:
        rng=np.random.default_rng(7); n=80
        frame={f:rng.normal(size=n) for f in get_features("A")}
        frame["velocity_kmh"]=2*frame["age"]+rng.normal(size=n)
        import pandas as pd
        frame=pd.DataFrame(frame); tr=frame.iloc[:60]; va=frame.iloc[60:]
        x,y,_=fit_transform_pair(tr,va,get_features("A"))
        model=make_ridge().fit(x,tr.velocity_kmh)
        print("Componentes sintéticos OK:",metrics(va.velocity_kmh,model.predict(y)))

if __name__=="__main__": main()
