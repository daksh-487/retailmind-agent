"""
api/index.py — FastAPI backend for RetailMind Product Intelligence Agent
Stateless design for Vercel serverless deployment.
"""

import sys
import os

# Ensure parent dir is on path for local imports
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_BASE_DIR = os.path.dirname(_THIS_DIR)
if _BASE_DIR not in sys.path:
    sys.path.insert(0, _BASE_DIR)

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import pandas as pd
import io
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="RetailMind API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_DIR = os.path.join(_BASE_DIR, "data")
PUBLIC_DIR = os.path.join(_BASE_DIR, "public")

def get_index_path():
    p1 = os.path.join(_THIS_DIR, "index.html")
    if os.path.exists(p1):
        return p1
    return os.path.join(PUBLIC_DIR, "index.html")



# ─── Dataset Loading ───────────────────────────────────────────────────────────

def load_dataset(products_csv: Optional[str] = None,
                 reviews_csv: Optional[str] = None):
    """Load products & reviews DataFrames — custom CSV strings or bundled samples."""
    try:
        if products_csv:
            products_df = pd.read_csv(io.StringIO(products_csv))
        else:
            products_df = pd.read_csv(os.path.join(DATA_DIR, "sample_products.csv"))

        if reviews_csv:
            reviews_df = pd.read_csv(io.StringIO(reviews_csv))
        else:
            reviews_df = pd.read_csv(os.path.join(DATA_DIR, "sample_reviews.csv"))

        return products_df, reviews_df
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Dataset error: {str(e)}")


# ─── RetailTools class (stateless, per-request) ────────────────────────────────

class RetailTools:
    def __init__(self, products_df: pd.DataFrame, reviews_df: pd.DataFrame, client: OpenAI):
        self.products_df = products_df
        self.reviews_df = reviews_df
        self.client = client
        self._review_cache: dict = {}

    # ── Tool 1 ───────────────────────────────────────────────────────────────
    def search_products(self, query: str, category: str = None) -> list:
        df = self.products_df.copy()
        if category and category.lower() != "all":
            df = df[df["category"].str.lower() == category.lower()]
        q = query.lower()

        def score(row):
            text = f"{row['product_name']} {row['category']}".lower()
            return sum(w in text for w in q.split())

        df["_score"] = df.apply(score, axis=1)
        df = df.sort_values("_score", ascending=False).head(5)
        return [
            {
                "product_id": row["product_id"],
                "product_name": row["product_name"],
                "category": row["category"],
                "price": float(row["price"]),
                "stock_quantity": int(row["stock_quantity"]),
                "avg_rating": float(row["avg_rating"]),
                "avg_daily_sales": float(row["avg_daily_sales"]),
            }
            for _, row in df.iterrows()
        ]

    # ── Tool 2 ───────────────────────────────────────────────────────────────
    def get_inventory_health(self, product_id: str) -> dict:
        df = self.products_df
        rows = df[df["product_id"] == product_id]
        if rows.empty:
            return {"error": f"Product {product_id} not found."}
        row = rows.iloc[0]
        stock = float(row["stock_quantity"])
        avg_sales = float(row["avg_daily_sales"])
        if avg_sales == 0:
            days = float("inf")
            status = "Healthy"
        else:
            days = round(stock / avg_sales, 1)
            status = "Critical" if days < 7 else ("Low" if days <= 14 else "Healthy")
        return {
            "product_id": product_id,
            "product_name": row["product_name"],
            "stock_quantity": int(stock),
            "avg_daily_sales": avg_sales,
            "days_to_stockout": days,
            "status": status,
            "reorder_level": int(row["reorder_level"]),
            "below_reorder": stock < row["reorder_level"],
        }

    # ── Tool 3 ───────────────────────────────────────────────────────────────
    def get_pricing_analysis(self, product_id: str) -> dict:
        df = self.products_df
        rows = df[df["product_id"] == product_id]
        if rows.empty:
            return {"error": f"Product {product_id} not found."}
        row = rows.iloc[0]
        price = float(row["price"])
        cost = float(row["cost"])
        margin = round((price - cost) / price * 100, 2)
        cat_avg = df[df["category"] == row["category"]]["price"].mean()
        pos = "Premium" if price >= cat_avg * 1.2 else ("Budget" if price <= cat_avg * 0.8 else "Mid-Range")
        return {
            "product_id": product_id,
            "product_name": row["product_name"],
            "price": price,
            "cost": cost,
            "gross_margin_pct": margin,
            "low_margin_flag": margin < 20,
            "price_positioning": pos,
            "category_avg_price": round(cat_avg, 2),
            "suggested_action": (
                f"Margin below 20% — consider raising price or reducing cost."
                if margin < 20 else "Margin is healthy."
            ),
        }

    # ── Tool 4 ───────────────────────────────────────────────────────────────
    def get_review_insights(self, product_id: str) -> dict:
        if product_id in self._review_cache:
            return self._review_cache[product_id]
        prow = self.products_df[self.products_df["product_id"] == product_id]
        if prow.empty:
            return {"error": f"Product {product_id} not found."}
        name = prow.iloc[0]["product_name"]
        revs = self.reviews_df[self.reviews_df["product_id"] == product_id]
        avg_r = float(prow.iloc[0]["avg_rating"])
        total = int(prow.iloc[0]["review_count"])
        if revs.empty:
            r = {"product_id": product_id, "product_name": name, "avg_rating": avg_r,
                 "total_reviews": total, "sentiment_summary": "No detailed reviews available.",
                 "positive_themes": [], "negative_themes": []}
            self._review_cache[product_id] = r
            return r
        texts = "\n".join(
            f"- [{r['rating']}*] {r['review_title']}: {r['review_text']}"
            for _, r in revs.iterrows()
        )
        prompt = (f'Analyse reviews for "{name}" and return JSON with keys: '
                  '"sentiment_summary" (2 sentences), "positive_themes" (top 2 short phrases), '
                  '"negative_themes" (top 2 short phrases).\n\nReviews:\n' + texts +
                  '\n\nReturn ONLY valid JSON, no markdown.')
        resp = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3, max_tokens=300,
        )
        try:
            parsed = json.loads(resp.choices[0].message.content)
        except Exception:
            parsed = {"sentiment_summary": resp.choices[0].message.content,
                      "positive_themes": [], "negative_themes": []}
        result = {"product_id": product_id, "product_name": name,
                  "avg_rating": avg_r, "total_reviews": total, **parsed}
        self._review_cache[product_id] = result
        return result

    # ── Tool 5 ───────────────────────────────────────────────────────────────
    def get_category_performance(self, category: str) -> dict:
        df = self.products_df
        cd = df if category.lower() == "all" else df[df["category"].str.lower() == category.lower()].copy()
        if cd.empty:
            return {"error": f"No products found for category: {category}"}
        cd = cd.copy()
        cd["gross_margin"] = (cd["price"] - cd["cost"]) / cd["price"] * 100
        cd["days_to_stockout"] = cd.apply(
            lambda r: r["stock_quantity"] / r["avg_daily_sales"] if r["avg_daily_sales"] > 0 else 999, axis=1)
        cd["daily_revenue"] = cd["price"] * cd["avg_daily_sales"]
        top3 = cd.nlargest(3, "daily_revenue")[["product_id", "product_name", "daily_revenue", "price"]].to_dict("records")
        return {
            "category": category,
            "total_skus": len(cd),
            "avg_rating": round(cd["avg_rating"].mean(), 2),
            "avg_margin_pct": round(cd["gross_margin"].mean(), 2),
            "total_stock_units": int(cd["stock_quantity"].sum()),
            "critical_stock_count": int((cd["days_to_stockout"] < 7).sum()),
            "low_stock_count": int(((cd["days_to_stockout"] >= 7) & (cd["days_to_stockout"] <= 14)).sum()),
            "top_3_revenue_products": [
                {"product_id": p["product_id"], "product_name": p["product_name"],
                 "daily_revenue_inr": round(p["daily_revenue"], 2), "price_inr": p["price"]}
                for p in top3
            ],
        }

    # ── Tool 6 ───────────────────────────────────────────────────────────────
    def generate_restock_alert(self, threshold_days: int = 7) -> list:
        df = self.products_df.copy()
        df["days_to_stockout"] = df.apply(
            lambda r: r["stock_quantity"] / r["avg_daily_sales"] if r["avg_daily_sales"] > 0 else 999, axis=1)
        at_risk = df[df["days_to_stockout"] <= threshold_days].sort_values("days_to_stockout")
        results = []
        for _, row in at_risk.iterrows():
            days = round(row["days_to_stockout"], 1)
            rev = float(row["price"]) * (float(row["stock_quantity"]) + float(row["avg_daily_sales"]) * threshold_days)
            results.append({
                "product_id": row["product_id"],
                "product_name": row["product_name"],
                "category": row["category"],
                "days_to_stockout": days,
                "stock_quantity": int(row["stock_quantity"]),
                "avg_daily_sales": float(row["avg_daily_sales"]),
                "status": "Critical" if days < 7 else "Low",
                "revenue_at_risk_inr": round(rev, 2),
            })
        return results

    def get_tool_map(self):
        return {
            "search_products": self.search_products,
            "get_inventory_health": self.get_inventory_health,
            "get_pricing_analysis": self.get_pricing_analysis,
            "get_review_insights": self.get_review_insights,
            "get_category_performance": self.get_category_performance,
            "generate_restock_alert": self.generate_restock_alert,
        }


# ─── Tool Schemas ──────────────────────────────────────────────────────────────

TOOL_SCHEMAS = [
    {"type": "function", "function": {
        "name": "search_products",
        "description": "Search for products in the StyleCraft catalog by text query and optional category.",
        "parameters": {"type": "object",
                       "properties": {
                           "query": {"type": "string", "description": "Search text"},
                           "category": {"type": "string",
                                        "enum": ["Tops","Dresses","Bottoms","Outerwear","Accessories","All"],
                                        "description": "Optional category filter"},
                       }, "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "get_inventory_health",
        "description": "Get inventory health for a product: stock, days to stockout, Critical/Low/Healthy status.",
        "parameters": {"type": "object",
                       "properties": {"product_id": {"type": "string"}},
                       "required": ["product_id"]}}},
    {"type": "function", "function": {
        "name": "get_pricing_analysis",
        "description": "Get pricing intelligence: gross margin %, price positioning, low-margin alerts.",
        "parameters": {"type": "object",
                       "properties": {"product_id": {"type": "string"}},
                       "required": ["product_id"]}}},
    {"type": "function", "function": {
        "name": "get_review_insights",
        "description": "Get AI-generated review insights: sentiment summary, positive and negative themes.",
        "parameters": {"type": "object",
                       "properties": {"product_id": {"type": "string"}},
                       "required": ["product_id"]}}},
    {"type": "function", "function": {
        "name": "get_category_performance",
        "description": "Get aggregated performance metrics for a category or all categories.",
        "parameters": {"type": "object",
                       "properties": {"category": {"type": "string",
                                                    "enum": ["Tops","Dresses","Bottoms","Outerwear","Accessories","All"]}},
                       "required": ["category"]}}},
    {"type": "function", "function": {
        "name": "generate_restock_alert",
        "description": "Scan all products at risk of stockout within threshold days, sorted by urgency.",
        "parameters": {"type": "object",
                       "properties": {"threshold_days": {"type": "integer", "default": 7}},
                       "required": []}}},
]

SYSTEM_PROMPT = """You are the RetailMind Product Intelligence Agent for StyleCraft, a D2C fashion brand.

You help the Product Manager answer questions about the product catalog in real time.

Tools available:
- search_products: Find products by text and/or category
- get_inventory_health: Stock levels, days to stockout, urgency status
- get_pricing_analysis: Gross margin %, price positioning, low-margin alerts
- get_review_insights: Customer sentiment, themes, ratings (LLM-generated)
- get_category_performance: Category-level aggregated metrics
- generate_restock_alert: Products at critical/low stock risk

Guidelines:
- Always use tools when data is needed — never guess from memory
- Present data clearly with Rs. for prices and % for margins
- Flag Critical stock items with a warning emoji
- If margin < 20%, suggest a pricing action
- If user asks about a product by name, first call search_products to find the ID
- Be concise but complete. Use bullet points for lists.
"""


# ─── Agent Chat (stateless) ────────────────────────────────────────────────────

def run_agent_chat(history: list, user_message: str, tools: RetailTools) -> tuple:
    """
    history: list of {role, content} — only user/assistant turns (no tool messages)
    Returns: (response_text, updated_history)
    """
    full_msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    for m in history:
        full_msgs.append({"role": m["role"], "content": m["content"]})
    full_msgs.append({"role": "user", "content": user_message})

    tool_map = tools.get_tool_map()

    while True:
        resp = tools.client.chat.completions.create(
            model="gpt-4o",
            messages=full_msgs,
            tools=TOOL_SCHEMAS,
            tool_choice="auto",
            temperature=0.4,
            top_p=0.95,
            max_tokens=1500,
        )
        msg = resp.choices[0].message
        reason = resp.choices[0].finish_reason

        if reason == "tool_calls" and msg.tool_calls:
            full_msgs.append(msg)
            for tc in msg.tool_calls:
                fn = tool_map.get(tc.function.name)
                args = json.loads(tc.function.arguments)
                result = fn(**args) if fn else {"error": f"Unknown tool: {tc.function.name}"}
                full_msgs.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(result, ensure_ascii=False),
                })
        else:
            final = msg.content or ""
            updated = history + [
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": final},
            ]
            return final, updated


# ─── Daily Briefing ────────────────────────────────────────────────────────────

def build_briefing(tools: RetailTools) -> str:
    df = tools.products_df
    lines = ["## Daily Briefing — StyleCraft Catalog\n"]

    alerts = tools.generate_restock_alert(threshold_days=14)
    critical = [a for a in alerts if a["days_to_stockout"] < 7]
    low = [a for a in alerts if 7 <= a["days_to_stockout"] <= 14]
    top3 = (critical + low)[:3]

    lines.append("### Critical Stock Alerts")
    if top3:
        for p in top3:
            icon = "🔴" if p["status"] == "Critical" else "🟡"
            lines.append(
                f"{icon} **{p['product_name']}** ({p['product_id']}) — "
                f"**{p['days_to_stockout']} days** left | "
                f"Revenue at risk: Rs.{p['revenue_at_risk_inr']:,.0f}"
            )
    else:
        lines.append("No critical stock alerts today.")

    lines.append("")
    worst = df.loc[df["avg_rating"].idxmin()]
    lines.append("### Lowest Rated Product")
    try:
        ins = tools.get_review_insights(worst["product_id"])
        summary = ins.get("sentiment_summary", "No reviews available.")
        neg = ", ".join(ins.get("negative_themes", [])) or "N/A"
        lines.append(
            f"🔻 **{worst['product_name']}** ({worst['product_id']}) — "
            f"Rating: **{worst['avg_rating']}**\n"
            f"> {summary}\n"
            f"> Key issues: _{neg}_"
        )
    except Exception:
        lines.append(f"🔻 **{worst['product_name']}** ({worst['product_id']}) — Rating: **{worst['avg_rating']}**")

    lines.append("")
    df2 = df.copy()
    df2["gross_margin"] = (df2["price"] - df2["cost"]) / df2["price"] * 100
    lm = df2[df2["gross_margin"] < 25].sort_values("gross_margin")
    lines.append("### Pricing Flag")
    if not lm.empty:
        p = lm.iloc[0]
        mg = round(p["gross_margin"], 1)
        lines.append(
            f"⚠️ **{p['product_name']}** ({p['product_id']}) — "
            f"Gross margin: **{mg}%** (below 25% threshold)\n"
            f"> Suggested: raise price by Rs.{round(p['price'] * 0.1):,} to improve margin."
        )
    else:
        lines.append("All products have healthy margins (>25%).")

    return "\n".join(lines)


# ─── Pydantic Models ───────────────────────────────────────────────────────────

class DatasetPayload(BaseModel):
    products_csv: Optional[str] = None
    reviews_csv: Optional[str] = None


class ChatRequest(BaseModel):
    messages: List[Dict[str, Any]] = []
    user_message: str
    products_csv: Optional[str] = None
    reviews_csv: Optional[str] = None
    category_filter: Optional[str] = None


# ─── API Endpoints ─────────────────────────────────────────────────────────────

def _get_client():
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY not configured")
    return OpenAI(api_key=key)


@app.get("/api/debug")
async def debug_paths():
    import os
    return {
        "this_dir": _THIS_DIR,
        "base_dir": _BASE_DIR,
        "public_dir": PUBLIC_DIR,
        "this_dir_contents": os.listdir(_THIS_DIR) if os.path.exists(_THIS_DIR) else [],
        "base_dir_contents": os.listdir(_BASE_DIR) if os.path.exists(_BASE_DIR) else [],
        "public_dir_contents": os.listdir(PUBLIC_DIR) if os.path.exists(PUBLIC_DIR) else [],
        "exists_index": os.path.exists(os.path.join(PUBLIC_DIR, "index.html")),
    }


@app.post("/api/stats")
async def get_stats(payload: DatasetPayload):
    products_df, _ = load_dataset(payload.products_csv, payload.reviews_csv)
    products_df = products_df.copy()
    products_df["gross_margin"] = (products_df["price"] - products_df["cost"]) / products_df["price"] * 100
    products_df["days_to_stockout"] = products_df.apply(
        lambda r: r["stock_quantity"] / r["avg_daily_sales"] if r["avg_daily_sales"] > 0 else 999, axis=1)
    return {
        "total_skus": len(products_df),
        "critical_skus": int((products_df["days_to_stockout"] < 7).sum()),
        "avg_margin": round(products_df["gross_margin"].mean(), 1),
        "avg_rating": round(products_df["avg_rating"].mean(), 2),
        "categories": sorted(products_df["category"].unique().tolist()),
    }


@app.post("/api/briefing")
async def get_briefing(payload: DatasetPayload):
    client = _get_client()
    products_df, reviews_df = load_dataset(payload.products_csv, payload.reviews_csv)
    tools = RetailTools(products_df, reviews_df, client)
    try:
        briefing = build_briefing(tools)
        return {"briefing": briefing}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/chat")
async def chat(req: ChatRequest):
    client = _get_client()
    products_df, reviews_df = load_dataset(req.products_csv, req.reviews_csv)
    tools = RetailTools(products_df, reviews_df, client)
    user_msg = req.user_message
    if req.category_filter and req.category_filter not in ("All Categories", "all", ""):
        user_msg = f"[Category filter: {req.category_filter}] {user_msg}"
    try:
        response, updated = run_agent_chat(req.messages or [], user_msg, tools)
        return {"response": response, "messages": updated}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/sample/products")
async def download_sample_products():
    path = os.path.join(DATA_DIR, "sample_products.csv")
    return FileResponse(path, media_type="text/csv",
                        headers={"Content-Disposition": "attachment; filename=sample_products.csv"})


@app.get("/api/sample/reviews")
async def download_sample_reviews():
    path = os.path.join(DATA_DIR, "sample_reviews.csv")
    return FileResponse(path, media_type="text/csv",
                        headers={"Content-Disposition": "attachment; filename=sample_reviews.csv"})


# ─── Serve frontend (local dev) ────────────────────────────────────────────────

@app.get("/")
async def serve_index():
    index = get_index_path()
    if os.path.exists(index):
        return FileResponse(index)
    return {"status": "RetailMind API running", "docs": "/docs"}


@app.get("/{full_path:path}")
async def serve_spa(full_path: str):
    # Don't intercept API routes
    if full_path.startswith("api/"):
        raise HTTPException(status_code=404)
    index = get_index_path()
    if os.path.exists(index):
        return FileResponse(index)
    raise HTTPException(status_code=404)
