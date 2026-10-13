from langchain_groq import ChatGroq
from dotenv import load_dotenv
from typing import TypedDict,Annotated,Optional

load_dotenv()

model=ChatGroq(
    model='openai/gpt-oss-120b',
    temperature=0.5
)

class Review(TypedDict):
    key_themes:Annotated[list[str],"Write down all the key themes discussed in the review in a list"]
    summary:Annotated[str,"A brief summary of the review under 100-150 words"]
    sentiment:Annotated[str,"Return sentiment either positive, negative or neutral and also give how much positive and how much negative from base line neutral as 0 to max 10"]
    pros:Annotated[Optional[list[str]],"Return pros about the review"]
    cons:Annotated[Optional[list[str]],"Return cons about the review"]

structured_model= model.with_structured_output(Review, method="json_schema", strict=True)

result=structured_model.invoke("""I bought MacBook Air M1 in 2021 Nov, from official apple website at price 93k and with student discount 83-84k something . I am working professional in IT industry. Have used Windows, Linux and Mac over the years and i thought it's good time to give review for those who would want to buy MacBook .
Let's start with Pros ( Bcz there are few )
Out of my 4 Laptop, macbook has long lasting battery, if i am doing heavy work, let's say video editing/coding it might last 2-3 hours, if surfing or watching online youtube, it might easily last around 7-8 hours even after 5 years, that's a big thing .

Any kind of designing or video editing software are kind of first citizen in Macbook, i have never faced any issue while using Adobe, Davinci Resolve or downloading any software from website, i remember when loading rendering heavy editing Davinci, it shutdown but it was rare case .

MacOS feels clean and beginner friendly, animations are also smooth but might still give 6 out of 10, will elaborate letter .

Best student laptop if you want to take notes, carry around in the University, College or joining for fest or whatever .

May be we can add more if i concentrate .

When i bought, i didn't have any BT device connected, mostly used wired headphones or handsfree. After warrenty completed, i bought buds, it had this weird connectivity issue, it would connect disconnect within 10-15 secs, it was cycle. the first i got to know during client meeting, it was my turn to speak and suddenly my buds were off . I complained to service center but they didn't acknowledge the issue as it was working fine with their Apple AirPods, came back home and tried again, same issue . Later on i got MagicMouse and Keyboard from friend, had same issue, at that point service centre resolved issue . Solution was to reinstall OS. i felt relief and enjoyed the buds, but when update came OS broke again and faced issue, only solution was to install OS again, wiping data and clean install . It was headache, i faced this issue till Sequoia came .

In my version, they only provide 2 USB C ports, 1 for charging and 1 for other usage, no HDMI, network socket or USB type nothing, you would have to buy C Hub . I always had external monitor, it was hassle because C Hub would wear out within year, especially having issue with HDMI cable . I would have to buy new one every 10-12 month .

I have done some settings, to personalise it, whenever there is major update, it would reset the setting and i would have to reconfigure them .

3rd party hardware connectivity is not reliable like one example about Bluetooth headphones, another one very popular is mouse, you will feel like scroll have glitch and jerks . There are only certain mouse that works well with Mac . Not to mention, their Magic mouse has horrible ergonomic design, i started to have palm aches as there is no hump to rest your hand .

This one has bothered me a lot during winter, if you have to work early morning or late night, due to winter bare metal body gets cold, it's hard to touch it . It takes 15-20 minutes time to warm up . I would have to run benchmark software to accelerate the process .

This one is also big, MacOS removed some kind of text bridge in their software, so in your external monitor, font doesn't look that much crisp . First i thought it was issue with monitor, later connected Linux and Windows with screen, everything worked well . Found on internet that a lot of people are facing same issue with text. There is work around for this but whenever there is an update, i will have to reconfigure it .

Mac os doesn't have much gaming support, there is limited options, hell even Linux has more games than Mac . So if you are thinking about gaming, forget Mac .

Over the years i have faced multiple basic issues with macos that i didn't expected it coming. i won't have bought it if i didn't have requirement of Mobile development . Recently they released new macOS tahoe and it has so may UI bugs, it feels like developer didn't bothered to polish/refine it before releasing . After taking the update, external monitor resolution went down, contacted support team but they couldn't help much . Had to rollback to older version .

To conclude, never buying mac or apple product again, and their support team also have this attitude like "We are apple" . I just want basic things to work . Giving 5/10 . Linux is still better option for me, never had to deal with so many issues . Share me if you had any similar issues or anything you would like to add .""")

print(result)