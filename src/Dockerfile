FROM public.ecr.aws/lambda/python:3.8

COPY cultu_codes_categories.npy model_cultures_only_2015_2019.pt requirements.txt ./

RUN python3.8 -m pip install -r requirements.txt -t .

COPY app.py ./

# Command can be overwritten by providing a different command in the template directly.
CMD ["app.lambda_handler"]