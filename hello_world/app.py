import json
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np


class CultuModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.pipe = nn.Sequential(
            nn.Linear(1150, 1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 287),
        )

    def forward(self, x):
        return self.pipe(x)


model = CultuModel()
model.load_state_dict(torch.load('./model_cultures_only_2015_2019.pt'))
model.eval()

cultu_categories = np.load('./cultu_codes_categories.npy', allow_pickle=True)[0]
cultu_code_to_index = {cultu_category: i for i, cultu_category in enumerate(cultu_categories)}

xy_mean = np.array([618013.6370889, 6635696.30789702])
xy_var = np.array([4.04535082e+10, 5.09237904e+10])

def lambda_handler(event, context):
    """
    Parameters
    ----------
    event: dict, required
        API Gateway Lambda Proxy Input Format

        Event doc: https://docs.aws.amazon.com/apigateway/latest/developerguide/set-up-lambda-proxy-integrations.html#api-gateway-simple-proxy-for-lambda-input-format

    context: object, required
        Lambda Context runtime methods and attributes

        Context doc: https://docs.aws.amazon.com/lambda/latest/dg/python-context-object.html

    Returns
    ------
    API Gateway Lambda Proxy Output Format: dict

        Return doc: https://docs.aws.amazon.com/apigateway/latest/developerguide/set-up-lambda-proxy-integrations.html
    """

    if event['httpMethod'] != 'POST':
        return {
            "statusCode": 405
        }

    print(event)

    body = json.loads(event['body'])
    cultures = body["cultures"]
    if len(cultures) != 4:
        return {
            "statusCode": 400,
            "body": "Expected exact 4 cultures codes"
        }
    try:
        cultu_indexes = torch.tensor([[cultu_code_to_index[culture] for culture in cultures]])
    except KeyError as e:
        return {
            "statusCode": 400,
            "body": f"Unexpected culture code: {e}"
        }
    cultu_indexes_one_hot = F.one_hot(cultu_indexes, num_classes=len(cultu_categories)).view(1, -1)
    coordinates = np.array([body['x'], body['y']])
    coordinates = torch.tensor((coordinates - xy_mean) / xy_var).view(1, -1)
    features = torch.cat((cultu_indexes_one_hot, coordinates), dim=1).float()
    prediction_one_hot = model(features)[0]
    prediction_index = prediction_one_hot.argmax().item()
    predicted_category = cultu_categories[prediction_index]

    return {
        "statusCode": 200,
        "body": json.dumps(
            {
                "prediction": predicted_category
            }
        ),
    }
