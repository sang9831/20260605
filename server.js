const express = require("express");
const { ChatOpenAI } = require("@langchain/openai");
const { ChatGroq } = require("@langchain/groq");
// const { ChatGoogleGenerativeAI } = require("@langchain/google-genai");
const { PromptTemplate } = require("@langchain/core/prompts");
const { StructuredOutputParser } = require("@langchain/core/output_parsers");
const dotenv = require("dotenv");
dotenv.config();
const { z } = require("zod");
const cors = require("cors");
const path = require("path");

const app = express();
// 3011포트가 server3.js용으로 사용 중이므로 게임 서버는 3012포트를 기본으로 사용합니다.
const PORT = 3012;

app.use(cors());
app.use(express.json({ limit: "50mb" }));
// 정적 파일 제공 (game.html을 서빙하기 위함, index.html 자동 매핑 방지)
app.use(express.static(path.join(__dirname), { index: false }));

// 1. 결과 스키마 정의 (JSON 파싱 자동화 - 위도/경도 제외)
const parser = StructuredOutputParser.fromZodSchema(
  z.object({
    country: z.string().describe("국가명"),
    region: z.string().describe("주/도/광역 지역명"),
    city: z.string().describe("도시명"),
    landmark: z.string().describe("가장 식별 가능한 랜드마크 또는 장소명"),
    confidence: z
      .number()
      .describe("위치 추정 신뢰도 0-100. 확실하면 90+, 추측이면 50 이하"),
    reason: z
      .string()
      .describe(
        "추론 근거: 식별한 시각적 단서 3가지 이상을 한국어로 구체적으로 서술",
      ),
  }),
);

// 2. 모델 설정 (Groq, NIM, OpenRouter)
const models = {
  groq: new ChatGroq({
    apiKey: process.env.GROQ_API_KEY,
    configuration: { baseURL: "https://api.groq.com/openai/v1" },
    model: "meta-llama/llama-4-scout-17b-16e-instruct",
  }),
  nim: new ChatOpenAI({
    apiKey: process.env.NIM_API_KEY,
    configuration: { baseURL: "https://integrate.api.nvidia.com/v1" },
    model: "meta/llama-4-maverick-17b-128e-instruct",
  }),
  openrouter: new ChatOpenAI({
    apiKey: process.env.OPENROUTER_API_KEY,
    configuration: { baseURL: "https://openrouter.ai/api/v1" },
    model: "qwen/qwen2.5-vl-72b-instruct",
  }),
};

// ─── 랜드마크 투표 앙상블 ──────────────────────────────────────

function normalizeLandmark(name) {
  return name
    .toLowerCase()
    .replace(/[^a-z0-9가-힣]/g, "") // 특수문자·공백 제거
    .trim();
}

function isSameLandmark(a, b) {
  const na = normalizeLandmark(a);
  const nb = normalizeLandmark(b);
  return na === nb || na.includes(nb) || nb.includes(na);
}

function ensembleLandmarkVoting(predictions) {
  if (!predictions || predictions.length === 0) {
    throw new Error("앙상블할 예측 결과가 없습니다.");
  }
  if (predictions.length === 1) {
    return {
      ...predictions[0].result,
      ensemble: { method: "single", votes: 1 },
    };
  }

  const groups = [];

  for (const pred of predictions) {
    const lm = pred.result.landmark || "";
    const existing = groups.find((g) => isSameLandmark(g.landmark, lm));

    if (existing) {
      existing.members.push(pred);
      existing.totalScore += pred.result.confidence;
    } else {
      groups.push({
        landmark: lm,
        members: [pred],
        totalScore: pred.result.confidence,
      });
    }
  }

  groups.sort(
    (a, b) =>
      b.members.length !== a.members.length
        ? b.members.length - a.members.length // 1차: 득표수
        : b.totalScore - a.totalScore, // 2차: 신뢰도 합
  );

  const winner = groups[0];
  const top = winner.members.reduce((a, b) =>
    a.result.confidence >= b.result.confidence ? a : b,
  );

  return {
    country: top.result.country,
    region: top.result.region,
    city: top.result.city,
    landmark: top.result.landmark,
    reason: top.result.reason,
    confidence: Math.round(winner.totalScore / winner.members.length),

    ensemble: {
      method: "landmark_voting",
      totalModels: predictions.length,
      votes: groups.map((g, i) => ({
        rank: i + 1,
        landmark: g.landmark,
        voteCount: g.members.length,
        totalScore: Math.round(g.totalScore),
        avgConfidence: Math.round(g.totalScore / g.members.length),
        models: g.members.map((m) => m.modelName),
      })),
    },
  };
}

async function analyze(name, model, base64, hint, history) {
  // 이전 대화 기록을 텍스트 프롬프트로 변환
  let historyText = "";
  if (history && history.length > 0) {
    historyText =
      `\n### 이전 시도 및 AI 예측 이력 (이전의 모든 오답 목록)\n` +
      history
        .map((h, idx) => {
          return `[시도 ${idx + 1}]
- 사용자가 준 추가 힌트/피드백: "${h.hint}"
- AI가 예측한 오답 정보: ${h.guess.country} ${h.guess.city} ${h.guess.landmark}
- 당시 AI의 추론 근거: ${h.guess.reason}`;
        })
        .join("\n\n") +
      "\n\n⚠️ **중요**: 위 시도들에 나온 랜드마크/장소는 모두 오답입니다. 절대 동일하거나 유사한 장소를 다시 정답으로 제시하지 마십시오. 새로운 힌트와 시각적 단서를 이용해 다른 가능성을 찾아야 합니다.";
  }

  const hintText = hint?.trim()
    ? `## 사용자 제공 최신 힌트\n${hint.trim()}\n최신 힌트와 이전 이력을 단서로 활용하되, 이미지의 시각적 분석을 최우선으로 하십시오.`
    : `## 사용자 제공 최신 힌트\n없음.`;

  const prompt = new PromptTemplate({
    template: `## 역할
당신은 이미지 한 장만으로 촬영 위치를 정확히 추정하는 전문 지리 분석가입니다.

## 미션
사용자가 올린 이미지와 추가 힌트(이전 오답 이력 포함)를 활용해, 이 이미지가 촬영된 정확한 랜드마크 또는 장소를 맞추십시오.

{history_text}

{hint_text}

## 언어 규칙
- country, region, city, landmark: 한국어로 표기하라. (예: "프랑스", "일드프랑스", "파리", "에펠탑")
- reason: 반드시 한국어로 서술하라.
- confidence: 숫자 그대로 유지.

## 출력 형식
{format_instructions}

규칙:
- 반드시 위 JSON 스키마만 출력하라. 마크다운 코드블록(\`\`\`), 설명 텍스트, 추가 키를 절대 포함하지 마라.
- 이전 오답 목록에 기록된 랜드마크나 장소명은 절대 최종 landmark로 출력하지 마십시오.
- landmark가 불분명하면 가장 가까운 유명 지점 이름을 한국어로 사용하라.`,
    inputVariables: [],
    partialVariables: {
      format_instructions: parser.getFormatInstructions(),
      hint_text: hintText,
      history_text: historyText,
    },
  });

  const formattedPrompt = await prompt.format({});

  const response = await model.invoke([
    {
      role: "user",
      content: [
        { type: "text", text: formattedPrompt },
        {
          type: "image_url",
          image_url: { url: `data:image/jpeg;base64,${base64}` },
        },
      ],
    },
  ]);

  const raw =
    typeof response.content === "string" ? response.content : response.content;
  const result = await parser.parse(raw);
  return { [name]: result };
}

app.post("/analyze", async (req, res) => {
  try {
    const { image, hint, history } = req.body;

    if (!image) {
      return res.status(400).json({ error: "이미지가 제공되지 않았습니다." });
    }

    const settled = await Promise.allSettled(
      Object.entries(models).map(([name, model]) =>
        analyze(name, model, image, hint, history),
      ),
    );

    const predictions = settled
      .filter((r) => r.status === "fulfilled")
      .map((r) => {
        const [modelName, result] = Object.entries(r.value)[0];
        return { modelName, result };
      });

    if (predictions.length === 0) {
      console.error(
        "실패 상세:",
        settled.map((r) => r.reason),
      );
      return res.status(502).json({
        error: "모든 모델이 실패했습니다. API 키나 모델 설정을 확인하세요.",
      });
    }

    const final = ensembleLandmarkVoting(predictions);

    res.json({
      final,
      details: predictions,
      failed: settled
        .filter((r) => r.status === "rejected")
        .map((r) => r.reason?.message),
    });
  } catch (err) {
    console.error("서버 에러:", err);
    res.status(500).json({ error: err.message });
  }
});

app.listen(PORT, function startServer() {
  console.log(`위치 추정 게임 서버가 포트 ${PORT} 에서 실행 중입니다.`);
  console.log(`접속 주소: http://localhost:${PORT}`);
});
