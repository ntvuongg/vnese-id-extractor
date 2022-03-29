from modules import *

class Yolo4(object):
    def get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    def load_yolo(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        self.class_names = self.get_class()
        self.anchors = self.get_anchors()

        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)

        # Generate colors for drawing bounding boxes.
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))

        self.sess = K.get_session()

        # Load model, or construct model and load weights.
        self.yolo4_model = yolo4_body(Input(shape=(608, 608, 3)), num_anchors//3, num_classes)
        self.yolo4_model.load_weights(model_path)

        print('{} model, anchors, and classes loaded.'.format(model_path))

        if self.gpu_num>=2:
            self.yolo4_model = multi_gpu_model(self.yolo4_model, gpus=self.gpu_num)

        self.input_image_shape = K.placeholder(shape=(2, ))
        self.boxes, self.scores, self.classes = yolo_eval(self.yolo4_model.output, self.anchors,
                len(self.class_names), self.input_image_shape,
                score_threshold=self.score)

    def __init__(self, score, iou, anchors_path, classes_path, model_path, gpu_num=1):
        self.score = score
        self.iou = iou
        self.anchors_path = anchors_path
        self.classes_path = classes_path
        self.model_path = model_path
        self.gpu_num = gpu_num
        self.load_yolo()

    def close_session(self):
        self.sess.close()

    def detect_image(self, image, model_image_size=(608, 608)):

        boxed_image = letterbox_image(image, tuple(reversed(model_image_size)))
        image_data = np.array(boxed_image, dtype='float32')

        print(image_data.shape)
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo4_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            })

        print('Found {} boxes for {}'.format(len(out_boxes), 'img'))

        font = ImageFont.truetype(font='font/FiraMono-Medium.otf',
                    size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness = (image.size[0] + image.size[1]) // 500 # Default is 300

        BB_file = open('bb_cor.txt', 'w')

        for i, c in list(enumerate(out_classes)):
            predicted_class = self.class_names[c]
            box = out_boxes[i]
            score = out_scores[i]

            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)

            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
            print(label, (left, top), (right, bottom))
            BB_file.write(str(c) + ' ' + str(left) + ' ' + str(top) + ' '  + str(right) + ' ' + str(bottom))
            if c < len(self.class_names) - 1:
              BB_file.write('\n')

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            # My kingdom for a good redistributable image drawing library.
            for i in range(thickness):
                draw.rectangle(
                    [left + i, top + i, right - i, bottom - i],
                    outline=self.colors[c])
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=self.colors[c])
            draw.text(text_origin, label, fill=(0, 0, 0), font=font)
            del draw

        BB_file.close()
        return image

os.environ['KMP_DUPLICATE_LIB_OK']='True'

ANCHORS = './yolo4_anchors.txt'
CORNER_MODEL = './models/weights/yolo4_corner_weight.h5'
CORNER_CLASSES = './models/classes/yolo_corner.names'
######
CONTENT_MODEL = './models/weights/yolo4_content_weight.h5'
CONTENT_CLASSES = './models/classes/yolo_content.names'
SCORE = 0.5
IOU = 0.5

model_image_size = (608, 608)
yolo4_corner_model = Yolo4(SCORE, IOU, ANCHORS, CORNER_CLASSES, CORNER_MODEL)
yolo4_content_model = Yolo4(SCORE, IOU, ANCHORS, CONTENT_CLASSES, CONTENT_MODEL)

app = Flask(__name__,template_folder='./')

@app.route("/", methods=['GET'])
def show_template():
    return render_template("./static/main.html")

UPLOAD_FOLDER = './uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

graph = tf.get_default_graph()
@app.route("/uploader", methods=['GET', 'POST'])
def upload():
     with graph.as_default():
        # print(request.files, flush=True)
        INPUT_IMG = os.listdir('./uploads')
        if INPUT_IMG is not None:
            for upload_img in INPUT_IMG:
                os.remove(os.path.join('./uploads', upload_img))

        if request.method == 'POST':
            # Get image from POST request
            f = request.files['file']
            # Save image to ./uploads
            f.save(os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(f.filename)))

            INPUT_IMG = os.listdir('./uploads')
            if INPUT_IMG is not None:
                img = Image.open(os.path.join('./uploads', INPUT_IMG[0]))
            else:
                print('Cant read image!')
            
            img_origin = img.copy()
            result = yolo4_corner_model.detect_image(img, model_image_size=model_image_size)
            result.save('./detection.jpg')
            
            total_bb = align.get_bb_cor('bb_cor.txt')

            if len(total_bb) < 4:
                response = {
                "data": "Missing corner! Detection failed!"
                }
            elif len(total_bb) > 4:
                response = {
                "data": "Corner noise detected! Detection failed!"
                }
                return jsonify(response), 404

            center_points = list(map(align.get_center_point, total_bb))
            # Temporary fixing
            c2, c3 = center_points[2], center_points[3]
            c2_fix, c3_fix = (c2[0],c2[1]+30), (c3[0],c3[1]+30)
            center_points = [center_points[0], center_points[1], c2_fix, c3_fix]
            center_points = np.asarray(center_points)
            aligned = align.four_point_transform(img_origin, center_points)
            aligned = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB)
            cv2.imwrite('./aligned.jpg', aligned)
            
            aligned_img = Image.open('./aligned.jpg')
            aligned_copy = aligned_img.copy()
            result2 = yolo4_content_model.detect_image(aligned_img, model_image_size=model_image_size)
            bb_cor = open('bb_cor.txt','r')
            if len(total_bb) < 10:
                response = {
                    "data": "Missing fields! Detection failed!"
                 }
            elif len(total_bb) > 10:
                response = {
                    "data": "Wrong fields detected! Detection failed!"
                 }
                return jsonify(response), 404
            bb_cor = [line.strip() for line in bb_cor]
            
            save_dir = os.path.join('./','static/src')
            if not os.path.isdir(save_dir):
                os.mkdir(save_dir)
            else:
                for f in os.listdir(save_dir):
                    os.remove(os.path.join(save_dir, f))

            for line in bb_cor:
                cord = line.split(' ')
                __, left, top, right, bottom = float(cord[0]), float(cord[1]), float(cord[2]), float(cord[3]), float(cord[4])
                cropped_image = aligned_copy.crop((left,top,right,bottom))
                cropped_image.save(os.path.join(save_dir, cord[0] + '.jpg'))

            """ Recognizion detected parts in ID """
            config = Cfg.load_config_from_name('vgg_transformer')
            config['weights'] = 'models/weights/transformerocr.pth'
            config['cnn']['pretrained']=False
            # config['device'] = 'cuda:0' # if using GPU
            config['device'] = 'cpu' # if using CPU
            config['predictor']['beamsearch']=False
            detector = Predictor(config)

            FIELDS_DETECTED = [] # Collecting all detected parts

            for img_crop in sorted(os.listdir(save_dir)):
                img_ = Image.open(os.path.join(save_dir,img_crop))
                s = detector.predict(img_)
                FIELDS_DETECTED.append(s)

            check_parts = [False] * 10

            for part in os.listdir(save_dir):
                if int(part[:-4]) in range(0,10):
                    check_parts[int(part[:-4])] = True

            if check_parts[7] is True:
                FIELDS_DETECTED = FIELDS_DETECTED[:7] + [FIELDS_DETECTED[7] + ' ' + FIELDS_DETECTED[8]] + [FIELDS_DETECTED[9]]
            else:
                FIELDS_DETECTED.pop(7)

            response = {
                "data": FIELDS_DETECTED
            }

            return jsonify(response)

if __name__ == "__main__":
    app.run(host='0.0.0.0',port='8080',debug=True)