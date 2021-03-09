from fastapi import FastAPI
from starlette.responses import JSONResponse
from pytorch import torch.load
import pandas as pd

app = FastAPI()

beer_select = torch.load('../models/pytorch_beer_selector.pt')

@app.get("/")
def read_root():
    return {"Hello": "Beers"}
	
@app.get('/health', status_code=200)
def healthcheck():
    return 'Get on the beers'
	
def format_features(brewery_name: str,	review_aroma:float, review_appearance:float, review_palate: float, review_taste: float):
  return {
        'Brewery Name': [brewery_name],
		'Aroma Score' : [review_aroma],
        'Appearance Score': [review_appearance],
        'Palate Score': [review_palate],
        'Taste Score': [review_taste]
    }
	
#@app.get("/brewery/beerselection")
#def predict(brewery_name: str,	review_aroma:float, review_appearance:float, review_palate: float, review_taste: float):
#    features = format_features(brewery_name, review_aroma, review_appearance, review_palate, review_taste)
#    obs = pd.DataFrame(features)
#    pred = beer_select.predict(obs)
#    return JSONResponse(pred.tolist())
	
