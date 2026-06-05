// 환경변수
const dotenv = require("dotenv");
dotenv.config();

// 의존성
const express = require("express");
const cors = require("cors");
// Provider
const { ChatGoogleGenerativeAI } = require("@langchain/google-genai");
const { ChatGroq } = require("@langchain/groq");
const { ChatOpenAI } = require("@langchain/openai");
// Core
const { PromptTemplate } = require("@langchain/core/prompts");
const {
  HumanMessage,
  SystemMessage,
  AIMessage,
} = require("@langchain/core/messages");

// 서버 구동
const PORT = process.env.PORT_04 ?? process.env.PORT ?? 3000;
const app = express();

// 미들웨어
app.use(cors()); // whitelist -> 어제 실습.
// 보안적으로는 최악의 선택 중 하나임을 일단은 명심.
app.use(express.json());
// npm i cors
// https://www.npmjs.com/package/cors

// 엔드포인트
app.post("/chat", async (req, res) => {
  console.log("[요청 해석]");
  const { provider, modelName, ask, history } = req.body;
  console.log("history", history);
  let model;
  switch (provider) {
    case "google-genai":
      model = await useGoogleGenAI(modelName);
      break;
    case "groq":
      model = await useGroq(modelName);
      break;
    case "nim":
      model = await useNim(modelName);
      break;
    default:
      throw new Error("지원하지 않는 Provider");
  }

  console.log("[프롬프트 포맷팅]");

  const promptTemplate = PromptTemplate.fromTemplate(
    "다음의 요청에 대해 친절하고 간결하게 응답 : {ask}",
  );
  const formattedPrompt = await promptTemplate.format({
    ask,
  });

  console.log("[모델 호출]");

  // System Instruction
  const mbti = "INTJ";
  const systemPromptTemplate = PromptTemplate.fromTemplate(
    "뒤의 성격을 의식하여 대답 : {mbti}",
  );
  const systemformattedPrompt = await systemPromptTemplate.format({
    mbti,
  });

  const prompts = [];

  // const response = await model.invoke([
  //   new SystemMessage(systemformattedPrompt),
  //   new HumanMessage(formattedPrompt),
  // ]);

  // 시스템 인스트럭션
  prompts.push(new SystemMessage(systemformattedPrompt));

  // 히스토리
  // 2가지
  // 1. 소화하는 히스토리 자체에 제약 -> 10개까지만.
  // for (const h of history) {
  for (const h of history.slice(
    Math.max(0, history.length - 10),
    history.length,
  )) {
    const { type, content } = h;
    prompts.push(
      type == "ask" ? new HumanMessage(content) : new AIMessage(content),
    );
  }
  // 2. compact -> 히스토리를 상대적으로 짧은 텍스트로 요약 (ai가 스스로)

  // 최종적 요청 사항
  prompts.push(new HumanMessage(formattedPrompt));

  console.log(prompts);

  const response = await model.invoke(prompts);

  console.log("[결과 정리]");

  //   console.log(response.text);

  console.log("[응답 전송]");

  res.json({
    answer: response.text,
  });
});

// 커스텀 함수
async function useGoogleGenAI(model) {
  // [Model]
  // gemini-3.1-flash-lite
  // gemma-4-26b-a4b-it // moe
  // gemma-4-31b-it // dense
  return new ChatGoogleGenerativeAI({
    apiKey: process.env.GEMINI_API_KEY,
    model,
    temperature: 0.7,
    // temperature: 0, // 고정된 응답의 형식을 바란다면 창의성/임의성을 의미하는 temperature는 최소로 (0)
    maxOutputTokens: 512,
    // json: true,
  });
}

async function useGroq(model) {
  // [Model]
  // openai/gpt-oss-20b // 빠름
  // openai/gpt-oss-120b // 생각 깊음 <- JSON Output을 목표로 하면 좀 생각이 깊은 모델 (패러미터가 높은...)
  // qwen/qwen3-32b // 추론형 모델 (thinking)
  // meta-llama/llama-4-scout-17b-16e-instruct
  return new ChatGroq({
    apiKey: process.env.GROQ_API_KEY,
    model,
    // temperature: 0,
    temperature: 0.7,
    maxOutputTokens: 512,
  });
}

async function useNim(model) {
  // [Model]
  // deepseek-ai/deepseek-v4-flash
  // deepseek-ai/deepseek-v4-pro
  // google/gemma-4-31b-it
  return new ChatOpenAI({
    apiKey: process.env.NIM_API_KEY,
    configuration: {
      baseURL: "https://integrate.api.nvidia.com/v1",
    },
    model,
    // temperature: 0,
    temperature: 0.7,
    maxOutputTokens: 512,
  });
}

// 리스너
app.listen(PORT, () => {
  console.log(`${PORT}에서 Listen 중`);
});
