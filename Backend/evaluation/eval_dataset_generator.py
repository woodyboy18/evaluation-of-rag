"""
Novel Recipe Evaluation Dataset Generator
Runs a benchmark suite of 10-15 novel synthesis queries through the existing RecipeFusion engine,
capturing the complete evaluation triplet:
  (User Query, Retrieved Parent Recipes, Generated Novel Recipe)
Saves the records into:
  1. Backend/memory/eval_novel_recipes.json (Primary evaluation dataset)
  2. Backend/memory/user_memory.json (Conversation memory)
"""

import os
import sys
import json
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from core.recipe_engine import answer_query
from memory.conversation_memory import save_turn, clear_history

# 30 Diverse Benchmark Queries for Novel Recipe Synthesis
BENCHMARK_NOVEL_QUERIES = [
    {
        "id": "novel_01",
        "query": "create a new high-protein vegetarian breakfast under 20 minutes without eggs",
        "category": "Dietary exclusion + High protein"
    },
    {
        "id": "novel_02",
        "query": "innovative vegan dinner combining chickpeas and spinach with no garlic",
        "category": "Vegan + Exclusion"
    },
    {
        "id": "novel_03",
        "query": "invent a unique low-carb skillet dish using paneer and broccoli",
        "category": "Low-carb + Pairing"
    },
    {
        "id": "novel_04",
        "query": "make a novel dessert with strawberry and oats without refined sugar",
        "category": "Dessert + Health constraint"
    },
    {
        "id": "novel_05",
        "query": "create a fusion dinner using tofu and peanut sauce without soy sauce",
        "category": "Fusion + Allergen exclusion"
    },
    {
        "id": "novel_06",
        "query": "create a healthy lunch bowl with lentils and sweet potato under 30 minutes",
        "category": "Time limit + Legume"
    },
    {
        "id": "novel_07",
        "query": "invent a savory breakfast dish combining avocado and mushrooms with no dairy",
        "category": "Dairy-free breakfast"
    },
    {
        "id": "novel_08",
        "query": "create a new recipe for chicken and quinoa without tomatoes",
        "category": "Poultry + Nightshade-free"
    },
    {
        "id": "novel_09",
        "query": "synthesize a creative dinner featuring salmon and asparagus with lemon butter",
        "category": "Seafood + Technique"
    },
    {
        "id": "novel_10",
        "query": "invent an Indian-Mediterranean fusion appetizer using paneer and olives",
        "category": "Cross-cuisine fusion"
    },
    {
        "id": "novel_11",
        "query": "create a quick 15-minute high-energy afternoon snack with dates and almonds",
        "category": "Quick snack + Energy"
    },
    {
        "id": "novel_12",
        "query": "synthesize a gluten-free vegetarian pasta substitute using zucchini and walnuts",
        "category": "Gluten-free + Vegetarian"
    },
    {
        "id": "novel_13",
        "query": "invent a spicy comforting winter soup using black beans and corn without onion",
        "category": "Soup + Allium-free"
    },
    {
        "id": "novel_14",
        "query": "create a novel dessert pairing mango and coconut milk with no gelatin",
        "category": "Vegan dessert + Tropical"
    },
    {
        "id": "novel_15",
        "query": "synthesize a creative high-fiber lunch using cauliflower rice and bell peppers",
        "category": "Keto/Fiber lunch"
    },
    {
        "id": "novel_16",
        "query": "invent a creative high-protein smoothie bowl using greek yogurt and blueberries without bananas",
        "category": "High-protein breakfast + Fruit exclusion"
    },
    {
        "id": "novel_17",
        "query": "create a spicy Mexican-Asian fusion dinner using chicken and kimchi",
        "category": "Cross-cuisine fusion + Poultry"
    },
    {
        "id": "novel_18",
        "query": "synthesize a diabetic-friendly low-carb dinner featuring cod and cauliflower mash",
        "category": "Diabetic-friendly + Seafood"
    },
    {
        "id": "novel_19",
        "query": "invent a dairy-free nut-free creamy pasta sauce using butternut squash and sunflower seeds",
        "category": "Allergen exclusion + Vegan pasta"
    },
    {
        "id": "novel_20",
        "query": "create a novel Mediterranean vegetarian wrap combining hummus, roasted eggplant, and pomegranate",
        "category": "Mediterranean + Vegetarian lunch"
    },
    {
        "id": "novel_21",
        "query": "synthesize a low-sodium dinner combining lean ground turkey and zucchini noodles without added salt",
        "category": "Low-sodium + Low-carb"
    },
    {
        "id": "novel_22",
        "query": "invent a refreshing summer salad pairing watermelon, cucumber, and fresh mint with no dairy",
        "category": "Dairy-free summer salad"
    },
    {
        "id": "novel_23",
        "query": "create a hearty vegan chili using sweet potatoes, kidney beans, and dark cocoa powder",
        "category": "Vegan stew + Creative pairing"
    },
    {
        "id": "novel_24",
        "query": "synthesize an Italian-Japanese fusion main dish featuring seared tuna and basil pesto",
        "category": "Cross-cuisine fusion + Seafood"
    },
    {
        "id": "novel_25",
        "query": "invent a gluten-free savory breakfast pancake using chickpea flour and spinach",
        "category": "Gluten-free + High-protein breakfast"
    },
    {
        "id": "novel_26",
        "query": "create an energizing snack bite combining chia seeds, rolled oats, and dark chocolate without peanuts",
        "category": "Nut-free energy snack"
    },
    {
        "id": "novel_27",
        "query": "synthesize a comforting autumn soup combining roasted carrots, ginger, and coconut milk",
        "category": "Vegan comforting soup"
    },
    {
        "id": "novel_28",
        "query": "invent a Mediterranean-style baked chicken breast stuffed with spinach and feta cheese",
        "category": "Low-carb Mediterranean dinner"
    },
    {
        "id": "novel_29",
        "query": "create a savory keto breakfast casserole using eggs, mushrooms, and bell peppers without cheese",
        "category": "Keto breakfast + Dairy-free"
    },
    {
        "id": "novel_30",
        "query": "synthesize a novel citrus dessert featuring lemon, chia pudding, and raspberries with no dairy",
        "category": "Vegan dessert + Fruit pairing"
    }
]

EVAL_OUTPUT_PATH = os.path.join(PROJECT_ROOT, "memory", "eval_novel_recipes.json")


def generate_benchmark_recipes(
    num_recipes: int = 15,
    novelty_level: str = "balanced",
    clear_existing_memory: bool = False
) -> str:
    """
    Executes novel recipe generation across benchmark queries and persists full triplets.
    """
    os.makedirs(os.path.dirname(EVAL_OUTPUT_PATH), exist_ok=True)
    if clear_existing_memory:
        clear_history(clear_eval=True)
        print("[Memory] Cleared conversation memory and evaluation dataset.")

    selected_queries = BENCHMARK_NOVEL_QUERIES[:num_recipes]
    
    print(f"\n{'='*70}")
    print(f"RECIPEFUSION NOVEL RECIPE DATASET GENERATOR")
    print(f"Generating {len(selected_queries)} novel recipes for LLM-as-a-Judge evaluation...")
    print(f"{'='*70}\n")

    eval_records = []

    for idx, item in enumerate(selected_queries, 1):
        q_id = item["id"]
        q_text = item["query"]
        category = item["category"]

        print(f"[{idx}/{len(selected_queries)}] Processing ({q_id}): \"{q_text}\"")
        start_time = time.time()

        try:
            result = answer_query(q_text, requested_mode="novel", novelty_level=novelty_level)
            elapsed = time.time() - start_time

            recipe_payload = {
                "title": result.get("title", "Untitled Novel Recipe"),
                "ingredients": result.get("ingredients", []),
                "instructions": result.get("instructions", []),
                "servings": result.get("servings", 2),
                "prep_time": result.get("prep_time", 10),
                "cook_time": result.get("cook_time", 15),
                "mode": result.get("mode", "synthesized")
            }

            retrieved = result.get("retrieved_recipes", [])
            constraints = result.get("constraints", {})

            eval_record = {
                "id": q_id,
                "user_query": q_text,
                "category": category,
                "constraints": constraints,
                "retrieved_recipes": retrieved,
                "generated_recipe": recipe_payload,
                "generation_time_sec": round(elapsed, 2),
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            }

            eval_records.append(eval_record)

            # Also update user_memory.json via conversation memory
            assistant_summary = f"{recipe_payload['title']} (synthesized)"
            save_turn(
                user_query=q_text,
                assistant_response=assistant_summary,
                constraints=str(constraints),
                recipe_details=recipe_payload,
                retrieved_recipes=retrieved
            )

            print(f"  -> Generated: \"{recipe_payload['title']}\" ({len(recipe_payload['ingredients'])} ings, {len(recipe_payload['instructions'])} steps)")
            print(f"  -> Retrieved Parents: {len(retrieved)} recipes | Latency: {round(elapsed, 2)}s\n")

        except Exception as e:
            print(f"  -> ERROR generating recipe for query '{q_text}': {e}\n")

    # Save to eval_novel_recipes.json
    with open(EVAL_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(eval_records, f, indent=4)

    print(f"{'='*70}")
    print(f"SUCCESS: Generated {len(eval_records)} novel recipes.")
    print(f"Saved evaluation dataset to: {EVAL_OUTPUT_PATH}")
    print(f"Saved conversation history to: {os.path.join(PROJECT_ROOT, 'memory', 'user_memory.json')}")
    print(f"{'='*70}\n")
    return EVAL_OUTPUT_PATH


if __name__ == "__main__":
    n = 15
    if len(sys.argv) > 1:
        try:
            n = int(sys.argv[1])
        except ValueError:
            pass
    generate_benchmark_recipes(num_recipes=n)
