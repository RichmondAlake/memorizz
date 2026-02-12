import asyncio
import os
import sys

import pytest
from dotenv import load_dotenv

# Add the project root to the Python path
project_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "src"))
load_dotenv()
os.environ.setdefault("MONGODB_URI", "mongodb://localhost:27017")

from memorizz.memagent import MemAgent  # noqa: E402
from memorizz.memory_provider.mongodb.provider import (  # noqa: E402
    MongoDBConfig,
    MongoDBProvider,
)
from scenario import Scenario, TestingAgent  # noqa: E402
from tests.mocks.mock_providers import MockLLMProvider, MockMemoryProvider  # noqa: E402

# Create a memory provider
use_real_mongo = os.environ.get("MEMORIZZ_TEST_REAL_MONGO") == "1"
if use_real_mongo:
    mongodb_config = MongoDBConfig(
        uri=os.environ["MONGODB_URI"], lazy_vector_indexes=True
    )
    memory_provider = MongoDBProvider(mongodb_config)
else:
    memory_provider = MockMemoryProvider()

Scenario.configure(testing_agent=TestingAgent(model="openai/gpt-4o-mini"))

mock_model = MockLLMProvider(
    [
        "Vegetarian recipe: Chickpea stir-fry.\n\nIngredients:\n- Chickpeas\n- Bell pepper\n- Onion\n- Garlic\n- Spinach\n\nSteps:\n1. Saute onion and garlic.\n2. Add peppers and chickpeas.\n3. Stir in spinach and serve."
    ]
)
mem_agent = MemAgent(memory_provider=memory_provider, model=mock_model)


@pytest.mark.agent_test
def test_vegetarian_recipe_agent():
    agent = mem_agent

    def vegetarian_recipe_agent(message, context):
        # Call your agent here
        response = agent.run(message)
        return {"message": response}

    # Define the scenario
    scenario = Scenario(
        "User is looking for a dinner idea",
        agent=vegetarian_recipe_agent,
        success_criteria=[
            "Recipe agent generates a vegetarian recipe",
            "Recipe includes a list of ingredients",
            "Recipe includes step-by-step cooking instructions",
        ],
        failure_criteria=[
            "The recipe is not vegetarian or includes meat",
            "The agent asks more than two follow-up questions",
        ],
    )

    # Run the scenario and get results
    result = asyncio.run(scenario.run())

    # Assert for pytest to know whether the test passed
    assert result.success
