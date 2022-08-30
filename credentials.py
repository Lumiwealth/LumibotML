import json


class AlpacaConfig:
    # Put your own Alpaca api key here:
    # API_KEY = "PK674RO5M858JZ217SPM"
    API_KEY = "PKFWJBFMSG24OVSCRVXA"
    # Put your own Alpaca secret here:
    API_SECRET = "QbxJmB0W3EFuE0AnZbc7LmqD0OJi4rijh6SqxgMd"
    API_SECRET = "uWm6opmroTTWuZ9Yr81XRTMsOMLNv8nvBmmLO4Dt"
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
