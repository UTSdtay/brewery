from fastapi import FastAPI
from starlette.responses import JSONResponse
import torch
import pandas as pd
from data_process import DataReader
from pytorch import PytorchMultiClass

app = FastAPI()
data_reader = DataReader()

beer_select = PytorchMultiClass(obs_tensor.shape[1])
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
	
@app.get("/brewery/beerselection")
def predict(brewery_name: str,	review_aroma:float, review_appearance:float, review_palate: float, review_taste: float):
    features = format_features(brewery_name, review_aroma, review_appearance, review_palate, review_taste)
    obs = pd.DataFrame(features)
    di = np.load('../src/models/brew_dict.npy',allow_pickle='TRUE').item()
    obs['brewery_name'].replace(di, inplace=True)
    obs_clean[num_cols] = data_reader.standard_scaler(obs_clean[num_cols])
    obs_clean.brewery_name=obs_clean.brewery_name.astype(int)
    obs_tensor = obs_clean.copy()
    obs_tensor = obs_tensor.to_numpy()
    obs_tensor = torch.from_numpy(obs_tensor)
    device = get_device()
    beer_select = PytorchMultiClass(obs_tensor.shape[1])
    beer_select.load_state_dict(torch.load('../app/src/models/pytorch_beer_selector.pt'))
    beer_select.eval()
    obs_tensor = obs_tensor.float()
    output = beer_select(obs_tensor).argmax(dim=1)
    target_encode = joblib.load('../src/models/target.joblib')
    pred = target_encode.inverse_transform(output)
    return JSONResponse(pred.tolist())