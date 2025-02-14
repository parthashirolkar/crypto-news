from pprint import pprint
from utils.get_bitoin_prices import get_data

from transformers import TextStreamer
from unsloth import FastLanguageModel
from templates.prompt_templates import prompt_style, question

max_seq_length = 2048
dtype = None
load_in_4bit = True


model, tokenizer = FastLanguageModel.from_pretrained(
    # model_name="unsloth/DeepSeek-R1-Distill-Llama-8B",
    model_name="finetuned/dalal-street-bro",
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)

streamer = TextStreamer(
    tokenizer,
    skip_prompt=True,
    skip_special_tokens=True,
    clean_up_tokenization_spaces=True,
)

publish_date = "2025-01-29"
title = "Why US tariff wars are taking a toll on bitcoin, other crypto"

news = """Major cryptocurrencies saw a slide on Monday, as a potential fallout of the fresh tariffs announced by the United States on some of its allies, with investors fearing an impending escalation in global wars and looking to move out of risky assets. Bitcoin was at a three-week low and ether at its lowest since September.

With a 25 per cent tariff on Canada and Mexico, and 10 per cent on China effective from Tuesday, US President Donald Trump has kicked off his promise of protectionist trade policies, triggering a fresh trade war with America’s top three trade partners that are also the largest contributors to its nearly $1 trillion trade deficit.

Since then, Canada has announced retaliatory tariffs on the US, triggering concerns of a full-blown trade war. China has said it will file a lawsuit against the tariffs. Trump has also signalled that new tariffs on the European Union will “definitely happen”. Meanwhile, the US has agreed to halt tariffs against Mexico for a month.

Why the impact on cryptocurrencies

Trump’s announcement of the tariffs came on a Saturday, which is a market holiday for most trading instruments, except for cryptocurrencies, which can be traded throughout the year. The fall in the digital currencies signifies that an uncertain trade environment could dent the rising crypto story, which has so far been the case under Trump’s Presidency.

Bitcoin fell to around $94,000 Monday morning, even touching a three-week low of $91,000. This is a significant fall after the coin touched a record high of more than $107,000 on January 20, when Trump was sworn in as President. Memecoins also bled profusely. One launched by Trump was down close to 12 per cent, and the coin launched by his wife Melania was down more than 13 per cent according to CoinMarketCap. Dogecoin was down more than 24 per cent over yesterday.
Festive offer

Meme coins are highly volatile cryptocurrency inspired by popular internet or cultural trends. They carry no intrinsic value but can soar, or plummet, in price. They are generally seen as indicators of retail investors’ interest in cryptocurrency. It is usually a marker of investors’ risk appetite at any given point in time.

Surge under Trump

Investors had predicted that Bitcoin could hit that mark if Trump were to be elected, since his entire campaign featured pro-crypto messaging, among other things. It was also anticipated that having Elon Musk — who has been a long time advocate for cryptocurrency — as a key adviser could further bolster investors’ belief in bitcoin, and other digital virtual assets, even if concerns around conflict of interest remain.
mail logo

Subscribe to receive the day's headlines from The Indian Express straight in your inbox
Some of Trump’s key picks for his administration, including Paul Atkins, to lead the Securities and Exchange Commission (SEC), and is widely considered a cryptocurrency advocate, also signalled a largely pro-crypto regulatory environment for the currency under the new administration."""


bitoin_data = get_data(end_date=publish_date)
news = news.replace("\n", " ")
news = news.replace("’", "'")


FastLanguageModel.for_inference(model).to("cuda")  # Optimizes the model for inference
inputs = tokenizer(
    [prompt_style.format(publish_date, title, news, bitoin_data, question, "")],
    return_tensors="pt",
).to("cuda")

_ = model.generate(
    input_ids=inputs.input_ids,
    streamer=streamer,
    attention_mask=inputs.attention_mask,
    max_new_tokens=1024,
    use_cache=True,
    repetition_penalty=1.1,
)

# response = tokenizer.batch_decode(outputs, clean_up_tokenization_spaces=True)[0]

# Extract only the response content
# if "### Response:" in response:
#     response = response.split("### Response:")[1].strip()


# json_match = re.search(r"\{.*\}", response, re.DOTALL)

# if json_match:
#     json_str = json_match.group(0)
#     json_str = re.sub(r"\s+", " ", json_str)

#     try:
#         data_dict = json.loads(json_str)
#     except json.JSONDecodeError as e:
#         print("JSON decoding failed:", e)
# else:
#     print("No JSON found in the text.")

# # Print with section headers for clarity
# print("\n" + "=" * 80)
# print("📢 **Generated Response** 📢\n")
# print(response)
# # pprint(data_dict, indent=4, sort_dicts=False)

# print("\n" + "=" * 80)
