"""
agent.py — LLM-powered Router Agent with conversation memory.
Uses OpenAI function-calling for tool dispatch. No keyword/regex routing.
"""

import json
import os
from openai import OpenAI
from tools import TOOL_SCHEMAS, dispatch_tool, _get_products, _get_reviews, get_review_insights, get_pricing_analysis, generate_restock_alert

# ── System prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are the RetailMind Product Intelligence Agent for StyleCraft, a D2C fashion brand with 80+ SKUs.

You help Priya Mehta (StyleCraft's Product Manager) answer questions about the product catalog in real time.

You have access to the following tools — always prefer using tools over answering from memory when data is needed:
- search_products: Find products by text and/or category
- get_inventory_health: Stock levels, days to stockout, urgency status
- get_pricing_analysis: Gross margin %, price positioning, low-margin alerts
- get_review_insights: Customer sentiment, themes, ratings (LLM-generated)
- get_category_performance: Category-level aggregated metrics
- generate_restock_alert: Products at critical/low stock risk

Routing rules (use LLM judgement, not keywords):
- INVENTORY queries → get_inventory_health or generate_restock_alert
- PRICING/MARGIN queries → get_pricing_analysis
- REVIEW/SENTIMENT queries → get_review_insights
- CATALOG/SEARCH queries → search_products or get_category_performance
- GENERAL/GREETING → answer conversationally using memory

Guidelines:
- Always present data clearly with ₹ for prices and % for margins
- Flag Critical stock items prominently with ⚠️
- If margin < 20%, suggest a pricing action
- Maintain context across turns — remember what was discussed
- Be concise but complete. Use bullet points for lists of products.
- If user asks about a product by name (not ID), first call search_products to find the ID, then call the relevant tool.
"""


class RetailMindAgent:
    """Stateful agent with conversation memory."""

    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.memory: list[dict] = []  # Full conversation history

    def reset(self):
        """Clear conversation memory."""
        self.memory = []

    def chat(self, user_message: str) -> str:
        """
        Process a user message, route to tools via LLM function-calling,
        and return the final response.
        """
        # Append user message to memory
        self.memory.append({"role": "user", "content": user_message})

        # Build messages: system + full history
        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + self.memory

        # ── Agentic loop: keep calling tools until LLM gives a final response ──
        while True:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                tools=TOOL_SCHEMAS,
                tool_choice="auto",   # Let LLM decide whether to call tools
                temperature=0.4,      # Moderate: factual but readable responses
                top_p=0.95,
                max_tokens=1500,
            )

            message = response.choices[0].message
            finish_reason = response.choices[0].finish_reason

            # If the LLM wants to call one or more tools
            if finish_reason == "tool_calls" and message.tool_calls:
                # Add assistant's tool-call message to history
                messages.append(message)

                # Execute each tool call
                for tool_call in message.tool_calls:
                    fn_name = tool_call.function.name
                    fn_args = json.loads(tool_call.function.arguments)

                    tool_result = dispatch_tool(fn_name, fn_args)
                    tool_result_str = json.dumps(tool_result, ensure_ascii=False)

                    # Append tool result to messages
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": tool_result_str,
                    })

            else:
                # LLM has given a final text response
                final_response = message.content or ""
                # Save assistant response to memory
                self.memory.append({"role": "assistant", "content": final_response})
                return final_response


# ── Daily Briefing Generator ──────────────────────────────────────────────────
def generate_daily_briefing(agent: RetailMindAgent) -> str:
    """
    Generate the startup Daily Briefing:
    - Top 3 most critically low-stock products
    - Worst-rated product + one-line reason
    - One pricing flag (lowest gross margin if < 25%)
    """
    df = _get_products()
    reviews_df = _get_reviews()

    lines = ["## 📋 Daily Briefing — StyleCraft Catalog\n"]

    # ── 1. Top 3 Critical Stock Products ─────────────────────────────────────
    alerts = generate_restock_alert(threshold_days=14)
    critical = [a for a in alerts if a["days_to_stockout"] < 7]
    low = [a for a in alerts if 7 <= a["days_to_stockout"] <= 14]
    top3_urgent = (critical + low)[:3]

    lines.append("### ⚠️ Critical Stock Alerts")
    if top3_urgent:
        for p in top3_urgent:
            status_icon = "🔴" if p["status"] == "Critical" else "🟡"
            lines.append(
                f"{status_icon} **{p['product_name']}** ({p['product_id']}) — "
                f"**{p['days_to_stockout']} days** left | "
                f"Revenue at risk: ₹{p['revenue_at_risk_inr']:,.0f}"
            )
    else:
        lines.append("✅ No critical stock alerts today.")

    lines.append("")

    # ── 2. Worst-Rated Product ────────────────────────────────────────────────
    worst = df.loc[df["avg_rating"].idxmin()]
    lines.append("### ⭐ Lowest Rated Product")

    # Get review insight for the worst rated product
    try:
        insights = get_review_insights(worst["product_id"])
        summary = insights.get("sentiment_summary", "No reviews available.")
        neg_themes = insights.get("negative_themes", [])
        neg_str = ", ".join(neg_themes) if neg_themes else "N/A"
        lines.append(
            f"🔻 **{worst['product_name']}** ({worst['product_id']}) — "
            f"Rating: **{worst['avg_rating']}★**\n"
            f"> {summary}\n"
            f"> Key issues: _{neg_str}_"
        )
    except Exception:
        lines.append(
            f"🔻 **{worst['product_name']}** ({worst['product_id']}) — "
            f"Rating: **{worst['avg_rating']}★** (no detailed reviews available)"
        )

    lines.append("")

    # ── 3. Pricing Flag (lowest gross margin < 25%) ───────────────────────────
    df2 = df.copy()
    df2["gross_margin"] = (df2["price"] - df2["cost"]) / df2["price"] * 100
    low_margin = df2[df2["gross_margin"] < 25].sort_values("gross_margin")

    lines.append("### 💰 Pricing Flag")
    if not low_margin.empty:
        p = low_margin.iloc[0]
        margin = round(p["gross_margin"], 1)
        lines.append(
            f"⚠️ **{p['product_name']}** ({p['product_id']}) — "
            f"Gross margin: **{margin}%** (below 25% threshold)\n"
            f"> Suggested action: Review cost structure or raise price by "
            f"₹{round(p['price'] * 0.1):,} to improve margin."
        )
    else:
        lines.append("✅ All products have healthy margins (>25%).")

    return "\n".join(lines)
