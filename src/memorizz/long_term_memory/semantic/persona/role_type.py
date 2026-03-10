# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

from enum import Enum


class RoleType(Enum):
    GENERAL = "General"
    ASSISTANT = "Virtual Assistant"
    CUSTOMER_SUPPORT = "Customer Support"
    TECHNICAL_EXPERT = "Technical Expert"
    RESEARCHER = "Researcher"


# Predefined default values for each role
PREDEFINED_INFO = {
    RoleType.GENERAL: {
        "goals": "Provide versatile support across various domains.",
        "background": "A general-purpose agent designed to adapt to multiple contexts.",
    },
    RoleType.ASSISTANT: {
        "goals": "Assist users by offering timely and personalized support.",
        "background": "An assistant agent crafted to manage schedules, answer queries, and help with daily tasks.",
    },
    RoleType.CUSTOMER_SUPPORT: {
        "goals": "Resolve customer issues promptly and provide clear guidance.",
        "background": "A customer support agent specialized in understanding user concerns and delivering effective solutions.",
    },
    RoleType.TECHNICAL_EXPERT: {
        "goals": "Provide expert technical advice and troubleshoot complex problems.",
        "background": "A technical expert agent with deep domain knowledge to assist with intricate technical issues.",
    },
    RoleType.RESEARCHER: {
        "goals": "Conduct thorough research and offer insights on advanced topics.",
        "background": "A researcher agent designed to synthesize complex information and present well-informed perspectives.",
    },
}
