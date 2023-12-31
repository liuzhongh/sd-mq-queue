import base64
import uuid
import oss2

from scripts.loadYamlFile import ExtraConfig


class ExtraFileStorage:
    def __init__(self, environment=None):
        config = ExtraConfig(environment)
        self.config_data = config.get_config()
        # 初始化客户端
        access_key_id = self.config_data['upload']['access_key']
        access_key_secret = self.config_data['upload']['secret_key']
        endpoint = self.config_data['upload']['server-addr']
        bucket_name = self.config_data['upload']['bucket_name']

        self.client = oss2.Bucket(oss2.Auth(access_key_id, access_key_secret), endpoint, bucket_name)

    def saveBase642Server(self, base64_data):
        # 解码base64数据为二进制数据
        binary_data = base64.b64decode(base64_data)

        # 生成不重复的文件名
        filename = str(uuid.uuid4()) + ".png"

        try:
            # 将二进制数据上传到Server
            self.client.put_object(filename, binary_data)

            # 返回访问路径
            return f"{filename}"
        except Exception as err:
            print(err)
            return None

    def saveBase64Files(self, images: []):
        rs = []
        if images:
            for img in images:
                url = self.saveBase642Server(img)
                rs.append(url)

        return rs

    def saveByte2Server(self, byte_data, file_extension):
        # 生成不重复的文件名
        filename = str(uuid.uuid4()) + '.' + file_extension

        try:
            # 将二进制数据上传到Server
            self.client.put_object(filename, byte_data)

            # 返回访问路径
            return f"{filename}"
        except Exception as err:
            print(err)
            return None

    def downloadFile(self, fileName):
        return self.client.get_object(fileName)
