from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(), override=True)


from loguru import logger
from flask import Flask, request
from utils import create_response
from flask_cors import CORS, cross_origin
from process import process_and_transform_image


###--------------------------------------------------------------------------###


app = Flask(__name__)
CORS(app)
app.config["CORS_HEADERS"] = "Content-Type"


###--------------------------------------------------------------------------###


def handle_request(func, *args, **kwargs):
    try:
        par = request.json if request.method == "POST" else request.args
        logger.info({"request": f"POST {request.path}", **par})
        return func(par, *args, **kwargs)
    except Exception as e:
        logger.error(e)
        return create_response(False, f"Error: An error occurred {e}", {}, 400)


###--------------------------------------------------------------------------###


@app.route("/transform_image", methods=["POST"])
@cross_origin()
def process_and_transform_image_route():
    return handle_request(process_and_transform_image)


###--------------------------------------------------------------------------###


def main():
    # Start the Flask app.
    app.run(host="0.0.0.0", port=8000, debug=False)


###--------------------------------------------------------------------------###


if __name__ == "__main__":
    main()
