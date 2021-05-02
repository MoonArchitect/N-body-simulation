#include <string>
#include <filesystem>

#include "NbodySystem.h"

using namespace Callbacks;
using namespace std;

BinaryDataSaver::BinaryDataSaver(int interval, string name, int maxFileMBSize) 
	: interval(interval), name(name), maxFileMBSize(maxFileMBSize * 1024 * 1024) {}

void BinaryDataSaver::reset(NbodySystem* system) {
	if (buffer != nullptr)
		delete[] buffer;
	int k = 3;// system->space == R3 ? 3 : 2;
	buffer = new float[system->N * k];
	fileSize = 0;
	fileIdx = 0;
	sizePerWrite = system->N * k * sizeof(float);

	filesystem::create_directory("data");

	file.open("data/masses.binary", ios_base::binary);

	for (int i = 0; i < system->N; i++)
		buffer[i] = system->host.pos_mass[i].w;

	file.write((char*)buffer, system->N * sizeof(float));
	file.close();

	file.open("data/data" + to_string(fileIdx++) + ".binary", ios_base::binary);
}

void BinaryDataSaver::start(int tick, NbodySystem* system) { }

void BinaryDataSaver::end(int tick, NbodySystem* system) {
	if (tick % interval == 0) {
		if (fileSize + sizePerWrite > maxFileMBSize){
			fileSize = 0;
			file.close();
			file.open("data/data" + to_string(fileIdx++) + ".binary", ios_base::binary);
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




CheckpointSaver::CheckpointSaver(int interval, string name)
	: interval(interval), name(name) {}


void CheckpointSaver::reset(NbodySystem* system) {
	if (buffer != nullptr)
		delete[] buffer;
	//int k = 3; // system->space == R3 ? 3 : 2;
	bufferSize = system->N;
	buffer = new float[bufferSize];

	filesystem::create_directory("configs/"); //  + name
	
	initConfig(system);
}

template <typename T>
void CheckpointSaver::writeJSON(string name, T value) {
	name = "\t\"" + name + "\": \"" + to_string(value) + "\",\n";
	file.write((char*)&name[0], name.size());
}

template <>
void CheckpointSaver::writeJSON <string> (std::string name, string value) {
	name = "\t\"" + name + "\": \"" + value + "\",\n";
	file.write((char*)&name[0], name.size());
}


void CheckpointSaver::initConfig(NbodySystem* system) {
	file.open("configs/" + name + ".json");
	file.write("{\n", 2);

	writeJSON("N", system->N);
	writeJSON("Space", string(system->space == R2 ? "R2" : "R3"));

	file.seekp(file.tellp() - 3ll);
	file.write("\n}", 2);
	file.close();
}

void CheckpointSaver::start(int tick, NbodySystem* system) { };

void CheckpointSaver::end(int tick, NbodySystem* system) {
	if ((tick + 1) % interval == 0) {
		file.open("configs/" + name + "_" + to_string(tick + 1) + ".params", ios_base::binary);

		system->updateHostData();
		save(system);

		file.close();
	}
}


void CheckpointSaver::save(NbodySystem* system) {
	int offset = 0;
	for (int n = 0; n < system->N; n++) {
		buffer[offset++] = system->host.pos_mass[n].x;
		buffer[offset++] = system->host.pos_mass[n].y;
		buffer[offset++] = system->host.pos_mass[n].z;
		buffer[offset++] = system->host.pos_mass[n].w;

		buffer[offset++] = system->host.vel[n].x;
		buffer[offset++] = system->host.vel[n].y;
		buffer[offset++] = system->host.vel[n].z;

		if (offset + 7 > bufferSize){
			file.write((char*)buffer, offset * sizeof(float));
			offset = 0;
		}
	}

	if (offset > 0)
		file.write((char*)buffer, offset * sizeof(float));
}


