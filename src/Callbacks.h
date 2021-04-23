#pragma once
#include <fstream>

class NbodySystem;

namespace Callbacks {

	class ICallback {
	public:
		virtual void reset() = 0;
		virtual void start() = 0;
		virtual void end() = 0;
	};


	class BinaryDataSaver {
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

}