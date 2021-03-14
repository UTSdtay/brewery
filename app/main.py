from fastapi import FastAPI
from starlette.responses import JSONResponse
import torch
import joblib
import pandas as pd
from src.models.data_process import DataReader
from src.models.pytorch import PytorchMultiClass

app = FastAPI()
data_reader = DataReader()

@app.get("/")
def read_root():
    return {"Hello": "Beers"}
	
@app.get('/health', status_code=200)
def healthcheck():
    return 'Get on the beers'
	
def format_features(brewery_name: str,	review_aroma:float, review_appearance:float, review_palate: float, review_taste: float):
  return {
        'brewery_name': [brewery_name],
        'review_aroma' : [review_aroma],
        'review_appearance': [review_appearance],
        'review_palate': [review_palate],
        'review_taste': [review_taste]
    }
	
@app.post("/type/beer")
def predict(brewery_name: str, review_aroma:float, review_appearance:float, review_palate: float, review_taste: float):
    features = format_features(brewery_name, review_aroma, review_appearance, review_palate, review_taste)
    obs = pd.DataFrame(features)
    brew_encode = joblib.load('../src/models/brewnames.joblib')
    obs['brewery_name']=brew_encode.transform(obs['brewery_name'])
    scale = joblib.load('../src/models/stdscale.joblib')
    obs[num_cols] = scale.transform(obs[num_cols])
    obs.brewery_name=obs.brewery_name.astype(int)
    obs = obs.to_numpy()
    obs = torch.from_numpy(obs)
    device = get_device()
    beer_select = PytorchMultiClass(obs.shape[1])
    beer_select.load_state_dict(torch.load('../src/models/pytorch_beer_selector.pt'))
    beer_select.eval()
    obs = obs.float()
    output = beer_select(obs).argmax(dim=1)
    target_encode = joblib.load('../src/models/target.joblib')
    pred = target_encode.inverse_transform(output)
    return JSONResponse(pred.tolist())

@app.post("/type/beers")
def predict(brewery_name: str,	review_aroma:float, review_appearance:float, review_palate: float, review_taste: float):
    features = format_features(brewery_name, review_aroma, review_appearance, review_palate, review_taste)
    obs = pd.DataFrame(features)
    brew_encode = joblib.load('../src/models/brewnames.joblib')
    obs['brewery_name']=brew_encode.transform(obs['brewery_name'])
    scale = joblib.load('../src/models/stdscale.joblib')
    obs[num_cols] = scale.transform(obs[num_cols])
    obs.brewery_name=obs.brewery_name.astype(int)
    obs = obs.to_numpy()
    obs = torch.from_numpy(obs)
    device = get_device()
    beer_select = PytorchMultiClass(obs.shape[1])
    beer_select.load_state_dict(torch.load('../src/models/pytorch_beer_selector.pt'))
    beer_select.eval()
    obs = obs.float()
    output = beer_select(obs).argmax(dim=1)
    target_encode = joblib.load('../src/models/target.joblib')
    pred = target_encode.inverse_transform(output)
    return JSONResponse(pred.tolist())

@app.post("/model/architecture/")
def print_model():
    beer_select = PytorchMultiClass(5)
    beer_select.load_state_dict(torch.load('../src/models/pytorch_beer_selector.pt'))
    return JSONResponse(beer_select.tolist())