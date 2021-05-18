FROM public.ecr.aws/lambda/python:3.7

COPY app.py cultu_codes_categories.npy model_cultures_only_2015_2019.pt requirements.txt ./

RUN python3.7 -m pip install -r requirements.txt -t .

# Command can be overwritten by providing a different command in the template directly.
CMD ["app.lambda_handler"]