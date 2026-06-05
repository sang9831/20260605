// 환경변수 불러오기
const dotenv = require("dotenv");
dotenv.config();

// 의존성
const express = require("express");

// 서버 세팅
const PORT = process.env.PORT ?? 3000;
const app = express();

app.listen(PORT, () => {
  console.log(`${PORT}에서 Listen 중`);
});

// npx nodemon
