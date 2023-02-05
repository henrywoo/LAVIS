import hiq
from hiq.framework.onnxruntime import OrtHiQLatency
# export TOKENIZERS_PARALLELISM=false

# _predict(image: UploadFile, question)
class MyFastAPI(hiq.HiQFastAPILatencyMixin, OrtHiQLatency):

    def __init__(self):
        super().__init__(extra_metrics={hiq.ExtraMetrics.ARGS})

    def custom(self):
        @self.inserter
        def predict(image, question):
            return self.o_predict(image, question)

        self.o_predict = hiq.mod("webapp")._predict
        hiq.mod("webapp")._predict = predict

        OrtHiQLatency.custom(self)

    def custom_disable(self):
        hiq.mod("webapp")._predict = self.o_predict
        OrtHiQLatency.custom_disable(self)


hiq.run_fastapi(driver=MyFastAPI(),
                app=hiq.mod("webapp").app,
                port=9090)





