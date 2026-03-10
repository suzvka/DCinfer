#pragma once

#include <stdexcept>
#include <string>
#include <sstream>

namespace DC {
	class Exception : public std::exception {
	public:
		enum class Level {
			Error,
			Warning,
			Info
		};

		Exception(
			const std::string& source = "Unknown",
			const std::string& message = "",
			Level level = Level::Error
		) : _level(level), _source(source) {
			_message = composeMessage(_level, _source, message);
		}

		Level getLevel() const noexcept { 
			return _level; 
		}

		const std::string& getSource() const noexcept { 
			return _source; 
		}

		const char* what() const noexcept override {
			return _message.c_str();
		}

	private:
		std::string _source;
		Level _level;

		std::string _message;

		static std::string composeMessage(
			Level level, 
			const std::string& source,
			const std::string& message
		) {
			std::ostringstream oss;
			oss << '[' << source << "] ";

			switch (level) {
			case Level::Error:   oss << "Error:";   break;
			case Level::Warning: oss << "Warning:"; break;
			case Level::Info:    oss << "Info:";    break;
			}

			if (!message.empty()) {
				oss << " - " << message;
			}
			
			return oss.str();
		}
	};
}