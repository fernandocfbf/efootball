import torch

class BallDetector():
    def __init__(self, threshold):
        model = torch.hub.load('ultralytics/yolov5', 'custom', path='efootball\src\classes\models\yolov5_ball\pre_trained_model.pt')
        self.predictor = model
        self.threshold = threshold

    def detectBalls(self, image):
        prediction = self.predictor(image)
        boxes = list()
        scores = list()
        if (len(prediction.pred[0]) > 0):
            score = prediction.pred[0][0][4]

            if float(score) >= float(self.threshold):
                x1 = int(prediction.pred[0][0][0])
                y1 = int(prediction.pred[0][0][1])
                x2 = int(prediction.pred[0][0][2])
                y2 = int(prediction.pred[0][0][3])
                boxes.append([y1,y2,x1,x2])
                scores.append(score)
        return {"boxes": boxes, "scores": scores}