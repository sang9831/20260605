// 환경변수 불러오기
const dotenv = require("dotenv");
dotenv.config();

// 의존성
const express = require("express");
// Provider
const { ChatGoogleGenerativeAI } = require("@langchain/google-genai");
const { ChatGroq } = require("@langchain/groq");
const { ChatOpenAI } = require("@langchain/openai");
// LangChain
const { HumanMessage } = require("@langchain/core/messages");
const { PromptTemplate } = require("@langchain/core/prompts");

// 서버 세팅
const PORT = process.env.PORT_02 ?? 3000;
const app = express();
// const googleGenAI = new ChatGoogleGenerativeAI();

// 미들웨어
app.use(express.json());

// 라우터, 엔드포인트 ...
app.post("/chat", async (req, res) => {
  console.log("요청 해석");
  const { provider, modelName, ask } = req.body;
  console.log(`provider : ${provider}`);
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

  console.log("프롬포트 포맷팅");

  const promptTemplate = PromptTemplate.fromTemplate(
    "당신은 MBTI가 {mbti}인 {job}입니다. 본인의 성격과 직업적 특징에 맞춰 뒤에 질문에 대답해주세요. {ask}",
  );
  const formattedPrompt = await promptTemplate.format({
    mbti: "INTJ",
    ask: ask,
    job: "부트캠프 강사",
  });

  console.log("모델 호출");

  const response = await model.invoke([new HumanMessage(formattedPrompt)]);

  console.log("결과 정리");
  console.log(response.text);
  console.log("응답 전송");
  res.json({
    answer: response.text,
  });
});

// 커스텀 함수
async function useGoogleGenAI(model) {
  return new ChatGoogleGenerativeAI({
    apiKey: process.env.GEMINI_API_KEY,
    model,
    temperature: 0.7,
    maxOutputTokens: 512,
  });
}

async function useGroq(model) {
  return new ChatGroq({
    apiKey: process.env.GROQ_API_KEY,
    model,
    temperature: 0.7,
    maxOutputTokens: 512,
  });
}
async function useNim(model) {
  return new ChatOpenAI({
    apiKey: process.env.NIM_API_KEY,
    configuration: {
      baseURL: "https://integrate.api.nvidia.com/v1",
    },
    model,
    temperature: 0.7,
    maxOutputTokens: 512,
  });
}

// 리스너
app.listen(PORT, () => {
  console.log(`${PORT}에서 Listen 중`);
});
