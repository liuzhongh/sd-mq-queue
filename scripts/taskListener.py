import base64
import json
import threading

from scripts.extLogging import logger
from scripts.mqEntry import MqSupporter

from scripts.fileStorage import ExtraFileStorage
from scripts.loadYamlFile import ExtraConfig
from modules.api import models
from modules.api.api import Api
from modules.call_queue import queue_lock
from pulsar import Message
from modules.shared import opts


def taskHandler(msg: Message, environment=None):
    from fastapi import FastAPI
    data = json.loads(msg.data())
    config = ExtraConfig(environment).get_config()

    if msg.topic_name() == config["queue"]["topic-t2i"]:
        txt2imgreq = models.StableDiffusionTxt2ImgProcessingAPI(**data)
        initData(txt2imgreq)
        logger.info("Text2Image Request '%s'", txt2imgreq)
        app = FastAPI()
        api = Api(app, queue_lock)
        response = api.text2imgapi(txt2imgreq)
        storage = ExtraFileStorage(environment)
        saveToStorage(storage, response)
        logger.info("Text2Image Result '%s'", response.dict())
        json_data = json.dumps(response.dict()).encode('utf-8')
        mq = MqSupporter(environment)
        mq.createProducer(config["queue"]["topic-t2i-result"], json_data, msg.properties())
        mq.createProducer(f"{config['queue']['topic-web-img-result']}-{msg.properties()['userId']}", json_data,
                          msg.properties())
    else:
        req = models.StableDiffusionImg2ImgProcessingAPI(**data)
        initData(req)
        logger.info("Image2Image Request '%s'", req)

        try:
            storage = ExtraFileStorage(environment)
            resp = storage.downloadFile(req.init_images[0])
            encoded_file = base64.b64encode(resp.read()).decode('utf-8')
            req.init_images = [encoded_file]
            app = FastAPI()
            api = Api(app, queue_lock)
            response = api.img2imgapi(req)
            saveToStorage(storage, response)
            logger.info("Image2Image Result '%s'", response.dict())
            json_data = json.dumps(response.dict()).encode('utf-8')
            mq = MqSupporter(environment)
            mq.createProducer(config["queue"]["topic-i2i-result"], json_data, msg.properties())
            mq.createProducer(f"{config['queue']['topic-web-img-result']}-{msg.properties()['userId']}", json_data,
                              msg.properties())
        except:
            logger.error("Image file download fail", exc_info=True)


def saveToStorage(storage, response):
    images = response.images
    if images is None:
        return

    image_array = []
    for i in range(len(images)):
        bytes_data = base64.b64decode(images[i])
        url = storage.saveByte2Server(bytes_data, opts.samples_format.lower())
        image_array.append(url)

    response.images = image_array


def handle_default():
    logger.info("Nothing to do")


def handle_roop(data):
    logger.info("roop %s:", data)
    data["args"][0] = "tsslfllweflwlfwf-ewfewf-ewfwefw-fwef-wfewfwef.png"


scripts_handle = {
    "roop": handle_roop
}


def initData(req):
    for alwayson_script_name in req.alwayson_scripts.keys():
        handler = scripts_handle.get(alwayson_script_name.lower(), handle_default)
        handler(req.alwayson_scripts[alwayson_script_name])


class TaskListener(threading.Thread):
    def __init__(self, environment=None):
        super().__init__()
        self.environment = environment

    def run(self):
        config = ExtraConfig(self.environment).get_config()
        mq = MqSupporter(self.environment)
        mq.createConsumer(config["queue"]["topics"], config["queue"]["subscription"], config["queue"]["consumer-name"],
                          taskHandler, self.environment)
        mq.closeClient()
