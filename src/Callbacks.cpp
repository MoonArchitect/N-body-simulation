#include <string>
#include <filesystem>

#include "NbodySystem.h"

using namespace Callbacks;

BinaryDataSaver::BinaryDataSaver(int interval, std::string name, int maxFileMBSize) 
	: interval(interval), name(name), maxFileMBSize(maxFileMBSize * 1024 * 1024) {}

void BinaryDataSaver::reset(NbodySystem* system) {
	if (buffer != nullptr)
		delete []buffer;
	int k = 3;// system->space == R3 ? 3 : 2;
	buffer = new float[system->N * k];
	fileSize = 0;
	fileIdx = 0;
	sizePerWrite = system->N * k * sizeof(float);

	std::filesystem::create_directory("data");

	file.open("data/masses.binary", std::ios_base::binary);

	for (int i = 0; i < system->N; i++)
		buffer[i] = system->host.pos_mass[i].w;

	file.write((char*)buffer, system->N * sizeof(float));
	file.close();

	file.open("data/data" + std::to_string(fileIdx++) + ".binary", std::ios_base::binary);
}

void BinaryDataSaver::start(int tick, NbodySystem* system) {
	
}

void BinaryDataSaver::end(int tick, NbodySystem* system) {
	if (tick % interval == 0) {
		if (fileSize + sizePerWrite > maxFileMBSize){
			fileSize = 0;
			file.close();
			file.open("data/data" + std::to_string(fileIdx++) + ".binary", std::ios_base::binary);
		}
		fileSize += sizePerWrite;

		system->updateHostData();
		save(system);
	}
}


void BinaryDataSaver::save(NbodySystem* system) {
	
	for (int i = 0; i < system->N; i++) {
		buffer[i * 3] = system->host.pos_mass[i].x;
		buffer[i * 3 + 1] = system->host.pos_mass[i].y;
		if (system->space == R3)
			buffer[i * 3 + 2] = system->host.pos_mass[i].z;
		else	
			buffer[i * 3 + 2] = 0;
	}
	file.write((char*)buffer, sizePerWrite);
}
