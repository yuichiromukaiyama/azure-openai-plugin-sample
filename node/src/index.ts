import "@tensorflow/tfjs-backend-cpu";
import { TensorFlowEmbeddings } from "langchain/embeddings/tensorflow";
import { FaissStore } from "langchain/vectorstores/faiss";

export const run = async () => {
  const vectorStore = await FaissStore.fromTexts(
    ["Hello world", "Bye bye", "hello nice world"],
    [{ id: 2 }, { id: 1 }, { id: 3 }],
    new TensorFlowEmbeddings()
  );

  const resultOne = await vectorStore.similaritySearch("hello world", 2);
  console.log(resultOne);
};

run().then(() => console.log("ok"));
