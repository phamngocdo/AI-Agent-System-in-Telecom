export const createUniqueConversationTitle = (conversations) => {
  const existingTitles = new Set(conversations.map(conv => conv.title));

  for (let i = 0; i < 100; i += 1) {
    const title = `Hội thoại ${Math.floor(1000 + Math.random() * 9000)}`;
    if (!existingTitles.has(title)) return title;
  }

  return `Hội thoại ${Date.now().toString().slice(-6)}`;
};
