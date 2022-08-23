import json


class AlpacaConfig:
    # Put your own Alpaca api key here:
    API_KEY = "PKXJ1U2OF2S048LOKX0D"
    # Put your own Alpaca secret here:
    API_SECRET = "CkFP7rvHJ9Yf4A70ImdLIEX74TWxK22lkWVoE15Q"
    # If you want to go live, you must change this
    ENDPOINT = "https://paper-api.alpaca.markets"

    def __init__(self, load_from_secret_manager=True):
        if load_from_secret_manager:
            from google.cloud import secretmanager

            # secret manager validation
            secret_id = "alpaca_lumibot"
            project_id = "arctic-plate-305019"
            client = secretmanager.SecretManagerServiceClient()
            id_name = client.secret_version_path(project_id, secret_id, "latest")

            # Access the secret version.
            id_response = client.access_secret_version(id_name)

            keys_dict = json.loads(id_response.payload.data.decode("utf-8"))

            # Put your own Alpaca key here:
            self.API_KEY = keys_dict["api_key_paper"]
            # Put your own Alpaca secret here:
            self.API_SECRET = keys_dict["secret_key_paper"]
            # If you want to go live, you must change this
            self.ENDPOINT = "https://paper-api.alpaca.markets"
        else:
            return None
