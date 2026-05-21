"""
tools.py — All 6 tool functions for the RetailMind Product Intelligence Agent.
Each function is callable by the LLM via OpenAI function-calling.
"""

import pandas as pd
import json
from functools import lru_cache
from openai import OpenAI
import os

# ── Load datasets once ────────────────────────────────────────────────────────
_products_df: pd.DataFrame | None = None
_reviews_df: pd.DataFrame | None = None


def _get_products() -> pd.DataFrame:
    global _products_df
    if _products_df is None:
        _products_df = pd.read_csv("retailmind_products.csv")
    return _products_df


def _get_reviews() -> pd.DataFrame:
    global _reviews_df
    if _reviews_df is None:
        _reviews_df = pd.read_csv("retailmind_reviews.csv")
    return _reviews_df


# ── Tool 1: search_products ───────────────────────────────────────────────────
def search_products(query: str, category: str = None) -> list[dict]:
    """
    Search and return matching products from the CSV based on a text query
    and optional category filter. Returns top 5 matches.
    """
    df = _get_products().copy()

    if category and category.lower() != "all":
        df = df[df["category"].str.lower() == category.lower()]

    query_lower = query.lower()
    # Score each row by how many query words appear in product_name or category
    def score(row):
        text = f"{row['product_name']} {row['category']}".lower()
        return sum(word in text for word in query_lower.split())

    df["_score"] = df.apply(score, axis=1)
    df = df[df["_score"] >= 0].sort_values("_score", ascending=False).head(5)

    results = []
    for _, row in df.iterrows():
        results.append({
            "product_id": row["product_id"],
            "product_name": row["product_name"],
            "category": row["category"],
            "price": float(row["price"]),
            "stock_quantity": int(row["stock_quantity"]),
            "avg_rating": float(row["avg_rating"]),
            "avg_daily_sales": float(row["avg_daily_sales"]),
        })
    return results


# ── Tool 2: get_inventory_health ──────────────────────────────────────────────
def get_inventory_health(product_id: str) -> dict:
    """
    Returns inventory status: current stock, avg daily sales,
    days to stockout, and a status flag (Critical/Low/Healthy).
    """
    df = _get_products()
    row = df[df["product_id"] == product_id]

    if row.empty:
        return {"error": f"Product {product_id} not found."}

    row = row.iloc[0]
    stock = float(row["stock_quantity"])
    avg_sales = float(row["avg_daily_sales"])

    # Guard against division by zero
    if avg_sales == 0:
        days_to_stockout = float("inf")
        status = "Healthy"
    else:
        days_to_stockout = round(stock / avg_sales, 1)
        if days_to_stockout < 7:
            status = "Critical"
        elif days_to_stockout <= 14:
            status = "Low"
        else:
            status = "Healthy"

    return {
        "product_id": product_id,
        "product_name": row["product_name"],
        "stock_quantity": int(stock),
        "avg_daily_sales": avg_sales,
        "days_to_stockout": days_to_stockout,
        "status": status,
        "reorder_level": int(row["reorder_level"]),
        "below_reorder": stock < row["reorder_level"],
    }


# ── Tool 3: get_pricing_analysis ──────────────────────────────────────────────
def get_pricing_analysis(product_id: str) -> dict:
    """
    Returns pricing intelligence: gross margin %, price positioning,
    and a low-margin flag if margin < 20%.
    """
    df = _get_products()
    row = df[df["product_id"] == product_id]

    if row.empty:
        return {"error": f"Product {product_id} not found."}

    row = row.iloc[0]
    price = float(row["price"])
    cost = float(row["cost"])
    category = row["category"]

    gross_margin = round((price - cost) / price * 100, 2)

    # Category average price for positioning
    cat_avg_price = df[df["category"] == category]["price"].mean()
    if price >= cat_avg_price * 1.2:
        positioning = "Premium"
    elif price <= cat_avg_price * 0.8:
        positioning = "Budget"
    else:
        positioning = "Mid-Range"

    return {
        "product_id": product_id,
        "product_name": row["product_name"],
        "price": price,
        "cost": cost,
        "gross_margin_pct": gross_margin,
        "low_margin_flag": gross_margin < 20,
        "price_positioning": positioning,
        "category_avg_price": round(cat_avg_price, 2),
        "suggested_action": (
            f"⚠️ Margin below 20% — consider raising price or reducing cost for {row['product_name']}."
            if gross_margin < 20
            else "Margin is healthy."
        ),
    }


# ── Tool 4: get_review_insights ───────────────────────────────────────────────
_review_cache: dict = {}


def get_review_insights(product_id: str) -> dict:
    """
    Uses an LLM to summarise customer reviews for a product.
    Returns avg rating, total reviews, 2-sentence sentiment summary,
    and top 2 recurring themes (positive and negative).
    """
    if product_id in _review_cache:
        return _review_cache[product_id]

    products_df = _get_products()
    reviews_df = _get_reviews()

    product_row = products_df[products_df["product_id"] == product_id]
    if product_row.empty:
        return {"error": f"Product {product_id} not found."}

    product_name = product_row.iloc[0]["product_name"]
    product_reviews = reviews_df[reviews_df["product_id"] == product_id]

    avg_rating = float(product_row.iloc[0]["avg_rating"])
    total_reviews = int(product_row.iloc[0]["review_count"])

    if product_reviews.empty:
        result = {
            "product_id": product_id,
            "product_name": product_name,
            "avg_rating": avg_rating,
            "total_reviews": total_reviews,
            "sentiment_summary": "No detailed reviews available in the dataset.",
            "positive_themes": [],
            "negative_themes": [],
        }
        _review_cache[product_id] = result
        return result

    # Concatenate review texts for LLM
    review_texts = "\n".join(
        f"- [{row['rating']}★] {row['review_title']}: {row['review_text']}"
        for _, row in product_reviews.iterrows()
    )

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    prompt = f"""You are a product analyst. Analyse these customer reviews for "{product_name}" and return a JSON object with:
- "sentiment_summary": a 2-sentence overall sentiment summary
- "positive_themes": list of top 2 positive recurring themes (short phrases)
- "negative_themes": list of top 2 negative recurring themes (short phrases)

Reviews:
{review_texts}

Return ONLY valid JSON, no markdown."""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,      # Low temp for factual summarisation
        max_tokens=300,
        top_p=0.9,
    )

    try:
        parsed = json.loads(response.choices[0].message.content)
    except Exception:
        parsed = {
            "sentiment_summary": response.choices[0].message.content,
            "positive_themes": [],
            "negative_themes": [],
        }

    result = {
        "product_id": product_id,
        "product_name": product_name,
        "avg_rating": avg_rating,
        "total_reviews": total_reviews,
        **parsed,
    }
    _review_cache[product_id] = result
    return result


# ── Tool 5: get_category_performance ─────────────────────────────────────────
def get_category_performance(category: str) -> dict:
    """
    Returns aggregated category-level metrics: total SKUs, avg rating,
    avg margin %, total stock units, low/critical stock count,
    and top 3 revenue-generating products.
    """
    df = _get_products()

    if category.lower() != "all":
        cat_df = df[df["category"].str.lower() == category.lower()]
    else:
        cat_df = df.copy()

    if cat_df.empty:
        return {"error": f"No products found for category: {category}"}

    cat_df = cat_df.copy()
    cat_df["gross_margin"] = (cat_df["price"] - cat_df["cost"]) / cat_df["price"] * 100
    cat_df["days_to_stockout"] = cat_df.apply(
        lambda r: r["stock_quantity"] / r["avg_daily_sales"] if r["avg_daily_sales"] > 0 else 999,
        axis=1,
    )
    cat_df["daily_revenue"] = cat_df["price"] * cat_df["avg_daily_sales"]

    critical_count = int((cat_df["days_to_stockout"] < 7).sum())
    low_count = int(((cat_df["days_to_stockout"] >= 7) & (cat_df["days_to_stockout"] <= 14)).sum())

    top3 = cat_df.nlargest(3, "daily_revenue")[
        ["product_id", "product_name", "daily_revenue", "price"]
    ].to_dict("records")

    return {
        "category": category,
        "total_skus": len(cat_df),
        "avg_rating": round(cat_df["avg_rating"].mean(), 2),
        "avg_margin_pct": round(cat_df["gross_margin"].mean(), 2),
        "total_stock_units": int(cat_df["stock_quantity"].sum()),
        "critical_stock_count": critical_count,
        "low_stock_count": low_count,
        "top_3_revenue_products": [
            {
                "product_id": p["product_id"],
                "product_name": p["product_name"],
                "daily_revenue_inr": round(p["daily_revenue"], 2),
                "price_inr": p["price"],
            }
            for p in top3
        ],
    }


# ── Tool 6: generate_restock_alert ────────────────────────────────────────────
def generate_restock_alert(threshold_days: int = 7) -> list[dict]:
    """
    Scans all products and returns those at risk of stockout within
    threshold_days, sorted by urgency. Includes estimated revenue at risk.
    """
    df = _get_products().copy()

    df["days_to_stockout"] = df.apply(
        lambda r: r["stock_quantity"] / r["avg_daily_sales"] if r["avg_daily_sales"] > 0 else 999,
        axis=1,
    )

    at_risk = df[df["days_to_stockout"] <= threshold_days].copy()
    at_risk = at_risk.sort_values("days_to_stockout")

    results = []
    for _, row in at_risk.iterrows():
        days = round(row["days_to_stockout"], 1)
        # Revenue at risk = price × (remaining stock + avg_daily_sales × threshold_days)
        revenue_at_risk = float(row["price"]) * (
            float(row["stock_quantity"]) + float(row["avg_daily_sales"]) * threshold_days
        )
        results.append({
            "product_id": row["product_id"],
            "product_name": row["product_name"],
            "category": row["category"],
            "days_to_stockout": days,
            "stock_quantity": int(row["stock_quantity"]),
            "avg_daily_sales": float(row["avg_daily_sales"]),
            "status": "Critical" if days < 7 else "Low",
            "revenue_at_risk_inr": round(revenue_at_risk, 2),
        })

    return results


# ── OpenAI Tool Schemas ────────────────────────────────────────────────────────
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "search_products",
            "description": "Search for products in the StyleCraft catalog by text query and optional category. Use for catalog discovery, finding specific products, or broad browsing.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search text, e.g. 'summer dress', 'blue top', 'winter jacket'",
                    },
                    "category": {
                        "type": "string",
                        "enum": ["Tops", "Dresses", "Bottoms", "Outerwear", "Accessories", "All"],
                        "description": "Optional category filter. Omit or use 'All' for all categories.",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_inventory_health",
            "description": "Get inventory health for a specific product: current stock, days to stockout, and status (Critical/Low/Healthy). Use when asked about stock levels, stockout risk, or inventory duration.",
            "parameters": {
                "type": "object",
                "properties": {
                    "product_id": {
                        "type": "string",
                        "description": "The product ID, e.g. SC001, SC012",
                    }
                },
                "required": ["product_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_pricing_analysis",
            "description": "Get pricing intelligence for a product: gross margin %, price positioning (Premium/Mid-Range/Budget), and low-margin alerts. Use for margin, profitability, or pricing queries.",
            "parameters": {
                "type": "object",
                "properties": {
                    "product_id": {
                        "type": "string",
                        "description": "The product ID, e.g. SC001, SC012",
                    }
                },
                "required": ["product_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_review_insights",
            "description": "Get AI-generated review insights for a product: sentiment summary, positive and negative themes. Use when asked about customer feedback, ratings, complaints, or sentiment.",
            "parameters": {
                "type": "object",
                "properties": {
                    "product_id": {
                        "type": "string",
                        "description": "The product ID, e.g. SC001, SC012",
                    }
                },
                "required": ["product_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_category_performance",
            "description": "Get aggregated performance metrics for a category or the whole catalog: total SKUs, average rating, average margin, stock health, and top revenue products.",
            "parameters": {
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "enum": ["Tops", "Dresses", "Bottoms", "Outerwear", "Accessories", "All"],
                        "description": "The category name, or 'All' for full catalog overview.",
                    }
                },
                "required": ["category"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "generate_restock_alert",
            "description": "Scan all products and return those at risk of stockout within a given number of days, sorted by urgency with revenue at risk. Use for restock alerts, urgent inventory issues, or stockout warnings.",
            "parameters": {
                "type": "object",
                "properties": {
                    "threshold_days": {
                        "type": "integer",
                        "description": "Number of days threshold for stockout risk. Default is 7.",
                        "default": 7,
                    }
                },
                "required": [],
            },
        },
    },
]

# ── Tool dispatcher ───────────────────────────────────────────────────────────
TOOL_MAP = {
    "search_products": search_products,
    "get_inventory_health": get_inventory_health,
    "get_pricing_analysis": get_pricing_analysis,
    "get_review_insights": get_review_insights,
    "get_category_performance": get_category_performance,
    "generate_restock_alert": generate_restock_alert,
}


def dispatch_tool(name: str, arguments: dict):
    """Execute a tool by name with given arguments."""
    fn = TOOL_MAP.get(name)
    if fn is None:
        return {"error": f"Unknown tool: {name}"}
    return fn(**arguments)
