# smart-intent-router
Smart intent router solution built on the Model Context Protocol (MCP). Features intent classification, multilingual support, dynamic LLM selection, and LM Studio integration, with both a web client and a modular server.


brew tap mongodb/brew
brew install mongodb-community@7.0
brew services start mongodb-community@7.0


Step 1: Install MongoDB on macOS and Windows
🧪 On macOS
Install with Homebrew

brew tap mongodb/brew
brew install mongodb-community@7.0

Start the service
brew services start mongodb-community@7.0

Check it runs
mongo --eval 'db.runCommand({ connectionStatus: 1 })'
You can use GUI like MongoDB Compass for easy inspection.