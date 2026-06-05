// 환경변수
const dotenv = require("dotenv");
dotenv.config();

// 의존성
const express = require("express");
// Provider
const { ChatGoogleGenerativeAI } = require("@langchain/google-genai");
const { ChatGroq } = require("@langchain/groq");
const { ChatOpenAI } = require("@langchain/openai");
// Core
const { PromptTemplate } = require("@langchain/core/prompts");
const { HumanMessage, SystemMessage } = require("@langchain/core/messages");

// 서버 구동
const PORT = process.env.PORT_03 ?? 3000;
const app = express();

// 미들웨어
app.use(express.json());

// 엔드포인트
app.post("/chat", async (req, res) => {
  console.log("[요청 해석]");
  const { provider, modelName, review } = req.body;
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

  const schema = {
    type: "object",
    propeties: {
      sentiment: {
        type: "string",
        enum: ["positive", "negative", "neutral"],
        description: "리뷰의 긍정, 부정, 중립 감정 분석 결과",
      },
      summary: {
        type: "string",
        description: "리뷰를 요약한 한국어 텍스트",
      },
    },
  };

  //   const promptTemplate = PromptTemplate.fromTemplate(
  //     "다음 고객 리뷰를 객관적으로 분석 : {review}, 뒤의 형식으로 구현, 마크다운 등으로 감싸지 말고 결과만 작성 : {schema}",
  //   );
  const promptTemplate = PromptTemplate.fromTemplate(
    "다음 고객 리뷰를 객관적으로 분석 : {review}",
  );
  const formattedPrompt = await promptTemplate.format({
    review,
    // schema,
  });

  console.log("[모델 호출]");

  // System Instruction
  const systemPromptTemplate = PromptTemplate.fromTemplate(
    "뒤의 형식으로 구현, 마크다운 등으로 감싸지 말고 결과만 작성. properties의 구성으로 json을 만들 것. : {schema}",
  );
  const systemformattedPrompt = await systemPromptTemplate.format({
    schema,
  });

  const response = await model.invoke([
    // new SystemMessage("대답은 질문과 상관없이 일본어를 사용함."),
    new SystemMessage(systemformattedPrompt),
    new SystemMessage("요약문의 경우에는 맨끝에 '냥'을 붙일 것."),
    new HumanMessage(formattedPrompt),
  ]);

  console.log("[결과 정리]");

  //   console.log(response.text);
  console.log(response.text);

  console.log("[응답 전송]");

  res.json({
    analysis: JSON.parse(response.text),
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
    // temperature: 0.7,
    temperature: 0, // 고정된 응답의 형식을 바란다면 창의성/임의성을 의미하는 temperature는 최소로 (0)
    // maxOutputTokens: 512,
    json: true,
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
    temperature: 0,
    // maxOutputTokens: 512,
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
    temperature: 0,
    // maxOutputTokens: 512,
  });
}

// 리스너
app.listen(PORT, () => {
  console.log(`${PORT}에서 Listen 중`);
});
