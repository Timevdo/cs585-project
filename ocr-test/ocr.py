from aspose.ocr import AsposeOcr, OcrInput, InputType, RecognitionSettings, DetectAreasMode

# Instantiate Aspose.OCR API
api = AsposeOcr()
# Add image to the recognition batch
input = OcrInput(InputType.SINGLE_IMAGE)
input.add("speed_template.png")
# Set document areas detection mode
recognitionSettings = RecognitionSettings()
recognitionSettings.detect_areas_mode = DetectAreasMode.TEXT_IN_WILD
# Recognize the image
result = api.recognize(input, recognitionSettings)
# Print recognition result
for i in range(len(result)):
    print(result[i].recognition_text)