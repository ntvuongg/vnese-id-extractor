from email import message
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import align

import os
import colorsys

import numpy as np
from keras import backend as K
from keras.layers import Input

from yolo4.model import yolo_eval, yolo4_body
from yolo4.utils import letterbox_image

import tensorflow as tf

from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg

from PIL import Image, ImageFont, ImageDraw
import cv2