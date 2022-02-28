from flask import Flask, json, request, jsonify, render_template
from BlindWaterMark import WaterMark
from base64 import b64encode

app = Flask(__name__)


@app.route('/BlindWaterMark/api/', methods=['POST', 'GET'])
def GetResult():
    if request.method == 'POST':
        SendMsg = {}
        RecvFile = request.files
        RecvForm = request.form

        if RecvForm.get('choice') == '0':
            RandomSeedWM = RecvForm.get('Seed1')
            RandomSeedDCT = RecvForm.get('Seed2')

            if (not RandomSeedWM.isdigit()) or (not RandomSeedDCT.isdigit()):
                SendMsg['status'] = 1
                SendMsg['msg'] = '格式错误'
                return jsonify(SendMsg)

            OriImg = RecvFile.get('OriImg')
            WMImg = RecvFile.get('WMImg')

            if OriImg is None or WMImg is None:
                SendMsg['status'] = 2
                SendMsg['msg'] = '文件上传失败'
                return jsonify(SendMsg)

            OriImg.save('./images/OriImg')
            WMImg.save('./images/WMImg')

            try:
                bwm = WaterMark(int(RandomSeedWM), int(RandomSeedDCT))
                bwm.ReadOriImg('./images/OriImg')
                bwm.ReadWM('./images/WMImg')
                bwm.Embed('./images/Embedded.png')
            except Exception:
                SendMsg['status'] = -1
                SendMsg['msg'] = '服务端异常'
                return jsonify(SendMsg)

            with open('./images/Embedded.png', 'rb') as file:
                SendMsg['OutImg'] = b64encode(file.read()).decode()

            SendMsg['status'] = 0
            SendMsg['msg'] = '嵌入水印成功'

            return SendMsg

        elif RecvForm.get('choice') == '1':
            RandomSeedWM = RecvForm.get('Seed1')
            RandomSeedDCT = RecvForm.get('Seed2')

            if (not RandomSeedWM.isdigit()) or (not RandomSeedDCT.isdigit()):
                SendMsg['status'] = 1
                SendMsg['msg'] = '格式错误'
                return jsonify(SendMsg)

            WMWidth = RecvForm.get('WMWidth')
            WMHeight = RecvForm.get('WMHeight')

            if (not WMWidth.isdigit()) or (not WMHeight.isdigit()):
                SendMsg['status'] = 1
                SendMsg['msg'] = '格式错误'
                return jsonify(SendMsg)

            EmbedImg = RecvFile.get('EmbedImg')

            if EmbedImg is None:
                SendMsg['status'] = 2
                SendMsg['msg'] = '文件上传失败'
                return jsonify(SendMsg)

            EmbedImg.save('./images/EmbedImg')

            try:
                bwm = WaterMark(int(RandomSeedWM), int(
                    RandomSeedDCT), (int(WMWidth), int(WMHeight)))
                bwm.Extract('./images/EmbedImg', './images/Extracted.png')
            except Exception:
                SendMsg['status'] = -1
                SendMsg['msg'] = '服务端异常'
                return jsonify(SendMsg)

            with open('./images/Extracted.png', 'rb') as file:
                SendMsg['OutImg'] = b64encode(file.read()).decode()

            SendMsg['status'] = 0
            SendMsg['msg'] = '提取水印成功'

            return SendMsg

        else:
            SendMsg['status'] = 1
            SendMsg['msg'] = '格式错误'
            return jsonify(SendMsg)
    else:
        return render_template('manual.html')


if __name__ == '__main__':
    app.run()
