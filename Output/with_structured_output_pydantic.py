from langchain_groq import ChatGroq
from pydantic import BaseModel, Field
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

model=ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.4
)

class Review(BaseModel):
    key_themes:list[str]=Field(description="Write down all the key themes discussed in the review in a list")
    summary:str=Field(description="A brief summary of the review under 100-150 words")
    sentiment:str=Field(description="Return sentiment either positive, negative or neutral and also give how much positive and how much negative from base line neutral as 0 to max 10")
    pros:Optional[list[str]]=Field(description="Return pros about the review")
    cons:Optional[list[str]]=Field(description="Return cons about the review")
    

structured_model=model.with_structured_output(Review, method="json_schema", strict=True)

result=structured_model.invoke("""I have nothing against Nolan as a director, and really enjoy some of his work. I walked into this expecting far more than what was delivered. Nolan needs to step away from the fantasy genre, stop writing his own scripts, and stick to his usual wheelhouse. His real strength has always been crafting massive atmosphere and visuals to support his scripts, but here only the atmosphere works at times, carried entirely by the sound design (except the soundtrack) and the cast.

The soundtrack is completely forgettable, which is unexpected for a Nolan film; not sure if it's due to the highly dynamic mix, but I honestly cannot remember a single tune or theme except the recurring motifs in the siren scene. Compare it to something like the soundtrack of Gladiator (2000) and the highly atmospheric focus of the soundtrack becomes apparent. The visuals also pale in comparison to Interstellar. It is baffling how Nolan and Hoyte van Hoytema managed to deliver such flat cinematography for an epic like this. The framing on standard IMAX was bizarre, locking roughly 90 percent of non-action scenes dead center while leaving empty negative space on the sides, almost as if it was meant to be seen only in 70mm IMAX, which I believe has an even taller aspect ratio. Paired with a bland color palette, the world fails to feel ancient, and at times feels more similar to the norse adaptations we've gotten like the series Vikings - especially the hellscape scene is basically an icelandic coast. Certain scenes look less like a movie and more like a stage play filled with people in cosplay. Nolan sticking to his usual approach simply doesn't work in a setting like this.

The script suffers just as much. The pacing unravels toward the end, and the dialogue - except for rare cases like Damon's monologue during the siren sequence - feels basic whenever it is actually audible over the mix. It's just highly functional surface level dialogue that only serves to advance Nolan's plots. This has become a habit of his at this point and I honestly don't get how these end up making it into the movie. His ego is showing there for sure.

The film does have its moments and is by no means a bad one. The Ithacans tumbling inside the Trojan Horse and the stormy sea sequences are visually impressive highlights. Beyond those, the action scenes were alright, but nothing really stuck with me, featuring no impressive camerawork or choreography. The witch hut was also a highlight as well where performances took over and crafted an immersive suspenseful atmosphere. I also enjoyed the siren song scene, where the dialogue actually had some depth and weight to it, and the soundtrack added to it.

Nolan's interpretation of Odysseus offers a compelling character arc, even if it is a massive departure from the source material where he is described as a general with massive hubris. The cast does heavy lifting as well: Matt Damon and John Leguizamo are great, Anne Hathaway delivers a strong performance, and Robert Pattinson is genuinely convincing in a slimy, snake-like role.

Still, a few good performances and few standout visuals just didn't do it for me. Instead of a refined, immersive epic, the result is an uninspired visual palette, repetitive center-weighted framing, and weak writing. I left the theater disappointed.""")

print(result)