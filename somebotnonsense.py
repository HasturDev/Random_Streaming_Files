import discord
import requests
import re
client = discord.Client()

#api key for Giphy p99LeIVnrws9GCWtbow9Oo4jNfJtBgac


@client.event
async def on_ready():
    print("Assuming direct Control as {0.user}".format(client))


@client.event
async def on_message(message):
    if message.author == client.user:
        return
        if message.author.bot: return
    
 #   if discord.MessageType.default:
 #       await message.channel.send("HOW DARE YOU NOT USE C# TO COMPLETE THIS PROJECT!!!")
    

    def transform(message):
        return message.content
    
    async for content in message.channel.history().map(transform):
        if message.author.bot: return
        if message.content.startswith('$meme'):
            memes = {'key': 'values'}
            response = requests.get("https://api.giphy.com/v1/gifs/search?api_key=p99LeIVnrws9GCWtbow9Oo4jNfJtBgac&q={}&limit=1&offset=&rating=R&lang=en".format(content), data = memes)
            if response.status_code == 200:
                print("success")
            else: print('errors')
            the_json = response.json()
            await message.channel.send(the_json.get('data')[0].get('url'))

    if message.content.startswith("Conanwhatisbestinlife"):
        await message.channel.send("https://lh3.googleusercontent.com/-7GTRpb3sKm0/W-8_dEWgYuI/AAAAAAAAsv4/0ty3KkE3P4IRd9Aty7p6ap-8ye5HPsRAwCHMYCw/Conan%2BThe%2BBarbarian%252C%2BWhat%2Bis%2Bbest%2Bin%2Blife%2Bquote%2BQC%2Bwm_thumb%255B4%255D?imgmax=800")
        await message.channel.send("To crush your enemies <br> some other stuff <br> meow")
        await message.channel.send("See them driven before you")
        await message.channel.send("And to hear the lamentation Of their women")


    if discord.SystemChannelFlags.join_notifications:
        await message.channel.send("DON'T TYPE THINGS")


    if message.content.startswith("$PythonWelcome"):
        await message.channel.send("Welcome to this server and enjoy our wonderful channels #general")


client.run('Token_Here')