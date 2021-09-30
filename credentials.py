from google.cloud import secretmanager
import json

class AlpacaConfig:
    API_KEY = ""
    # Put your own Alpaca secret here:
    API_SECRET = ""
    # If you want to go live, you must change this
    ENDPOINT = ""

    def __init__(self):
        # secret manager validation
        #secret_id  = 'Alpaca_Key-Rob'
        secret_id  = 'alpaca_lumibot'
        project_id = 'arctic-plate-305019'
        client  = secretmanager.SecretManagerServiceClient()
        id_name = client.secret_version_path(project_id, secret_id, "latest")


        # Access the secret version.
        id_response = client.access_secret_version(id_name)

        keys_dict = json.loads(id_response.payload.data.decode("utf-8"))
        
        # Put your own Alpaca key here:
        self.API_KEY = keys_dict['api_key_paper']
        # Put your own Alpaca secret here:
        self.API_SECRET = keys_dict['secret_key_paper']
        # If you want to go live, you must change this
        self.ENDPOINT = "https://paper-api.alpaca.markets"
