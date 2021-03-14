from fastapi import FastAPI
from starlette.responses import JSONResponse
import torch
import joblib
import pandas as pd
from src.models.data_process import DataReader
from src.models.pytorch import PytorchMultiClass

app = FastAPI()
data_reader = DataReader()

beer_select = PytorchMultiClass()
beer_select.load_state_dict(torch.load('../app/src/models/pytorch_beer_selector.pt'))

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
	
@app.get("/beerselection")
def predict(brewery_name: str,	review_aroma:float, review_appearance:float, review_palate: float, review_taste: float):
    features = format_features(brewery_name, review_aroma, review_appearance, review_palate, review_taste)
    obs = pd.DataFrame(features)
    di = np.load('../src/models/brew_dict.npy',allow_pickle='TRUE').item()
    obs['brewery_name'].replace(di, inplace=True)
    obs[num_cols] = data_reader.standard_scaler(obs[num_cols])
    obs.brewery_name=obs.brewery_name.astype(int)
    obs = obs.to_numpy()
    obs = torch.from_numpy(obs)
    device = get_device()
    beer_select = PytorchMultiClass(obs.shape[1])
    beer_select.load_state_dict(torch.load('../app/src/models/pytorch_beer_selector.pt'))
    beer_select.eval()
    obs = obs.float()
    output = beer_select(obs).argmax(dim=1)
    target_encode = joblib.load('../src/models/target.joblib')
    pred = target_encode.inverse_transform(output)
    return JSONResponse(pred.tolist())