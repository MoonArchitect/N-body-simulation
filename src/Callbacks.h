#pragma once
#include <fstream>

class NbodySystem;

namespace Callbacks {

	class Callback {
	public:
		virtual void reset(NbodySystem* system) = 0;
		virtual void start(int tick, NbodySystem* system) = 0;
		virtual void end(int tick, NbodySystem* system) = 0;
	};


	class BinaryDataSaver : public Callback {
		int maxFileMBSize, interval, fileIdx;
		float* buffer = nullptr, fileSize, sizePerWrite;
		std::ofstream file;
		std::string name;

	public:
		BinaryDataSaver(int interval, std::string name = "", int maxFileMBSize = 500);

		void reset(NbodySystem* system);
		void start(int tick, NbodySystem* system);
		void end(int tick, NbodySystem* system);

		void save(NbodySystem* system);
	};

	class CheckpointSaver : public Callback {
		int interval, bufferSize;
		float* buffer = nullptr;
		std::ofstream file;
		std::string name;

	public:
		CheckpointSaver(int interval, std::string name = "default");

		void reset(NbodySystem* system);
		void start(int tick, NbodySystem* system);
		void end(int tick, NbodySystem* system);

		void save(NbodySystem* system);
		void initConfig(NbodySystem* system);
		
		template <typename T>
		void writeJSON (std::string name, T value);
		
		template <>
		void writeJSON <std::string> (std::string name, std::string value);
	};

}