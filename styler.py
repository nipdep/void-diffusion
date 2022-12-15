
def prompt_styler(prompts):
  # with open(style_config, 'r') as pf:
  #   config = json.load(pf)
  # print(config)
  config = {
    "main": [
      "a painting of",
      "a digital art of", 
      "a digital painting of"
    ],
    "artist": [
      "by Charlie Bowater",
      "by Bastien Lecouffe-Deharme",
      "by Yoshitaka Amano",
      "by Karol Bak ",
      "by Yoann Lossel",
      "by Peter Mohrbacher",
      "by Ryohei Hase",
      "by Lü Ji",
      "by tooth wu",
      "by greg rutkowski",
      "by gaston bussiere",

      "by yoshitaka amano", 
      "by tsutomu nihei", 
      "by donato giancola", 
      "by tim hildebrandt"
    ],
    "style": [
      "Cgsociety",
      "Poster art",
      "Digital painting",
      "Steampunk",
      "Gothic art",
      "Dystopian art",
      "Fantasy art",
      "Matte painting",
      "Cinematic lighting",
      "focus , sharp",
      "huge scene",
      "unreal 5",
      "hyper realism",
      "ultra detailed fantasy",
      "lotr game design fanart by concept art",
      "devinart",
      "concept art",
      "fantasy character portrait",
      "extreme detail"
    ]
  }
  # filter any random artist
  prompts = [p[:p.find('by')] for p in prompts]

  new_prompts = [
      "{main} {i}, {artist}, {style}, detailed, pixiv".format(main=choice(config['main']), i=i, artist=', '.join(sample(config['artist'], 3)), style=', '.join(sample(config['style'], 3)))
      for i in prompts
  ]
  return new_prompts
