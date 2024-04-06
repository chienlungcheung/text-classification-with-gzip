import numpy as np
import gzip
from typing import Optional

# KnnClassifier 定义
class KnnClassifier:
    def __init__(self,
                 training_data: np.array,
                 training_labels: Optional[np.ndarray] = None,
                 top_k: Optional[int] = 1
                 ) -> None:
        self.training_data = training_data
        self.training_labels = training_labels
        self.compressor = gzip
        self.top_k = top_k
        return

    # 预测 x 与训练集哪个类型比较相配，以及距离如何
    def predict(self, x: str) -> tuple[np.array, np.array]:
        distances: list = []
        # 遍历训练集，计算 x 与每个训练集每个元素的距离
        for y in self.training_data:
            # append 保留了 y 在 training_data 中的顺序
            distances.append(self._ncd(x, y))
        # 按照从小到大对 distances 元素排序，记录从小到大元素对应的索引
        distances_np = np.array(distances)
        sorted_idx: np.array = np.argsort(distances_np)
        if self.top_k > len(sorted_idx):
            self.top_k = len(sorted_idx)
        targets = sorted_idx[:self.top_k]
        return self.training_data[targets], distances_np[targets]

    # 计算两个字符串的 normalized compression distance（ncd）
    # NCD 的值介于 0 和 1 之间，0 表示两个字符串完全相同，1 表示两个字符串完全不同
    def _ncd(self, x: str, y: str) -> float:
        Cx: int = len(self.compressor.compress(x.encode("utf-8")))
        Cy: int = len(self.compressor.compress(y.encode("utf-8")))
        Cxy: int = len(self.compressor.compress(" ".join([x, y]).encode("utf-8")))
        return (Cxy - min(Cx, Cy)) / max(Cx, Cy)

def main():
    training_data = [" #39;Jackal #39; Accomplice Acquitted of Bombings",
                     " #39;Marathon #39; mice bred in genetic first in US",
                     "Pilots Could Control Fate of US Airways"]
    test_data = [" #39;Jackal aide #39; in case 2 acquittal",
                 " #39;Marathon #39; mice engineered for extra endurance",
                 "Businesses Plan Attack on Edwards"]

    model = KnnClassifier(np.array(training_data), top_k=1)
    for x in test_data:
        nearest_class, distance = model.predict(x)
        print("test:[", x, "]\t top_k_class:[", nearest_class[0], "]\t distance:", distance[0])
if __name__ == "__main__":
        main()